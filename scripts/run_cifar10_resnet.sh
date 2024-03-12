nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm FedAvg --dataset Cifar10 --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --gpu 0 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm PerAvg --dataset Cifar10 --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --gpu 2 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm pFedMe --dataset Cifar10 --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --gpu 3 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm FedAvg --dataset Cifar10 --num_global_iters 1000 --aggr MtoSFed --times 1 --gamma 1 --gpu 4 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm PerAvg --dataset Cifar10 --num_global_iters 1000 --aggr MtoSFed --times 1 --gamma 1 --gpu 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm pFedMe --dataset Cifar10 --num_global_iters 1000 --aggr MtoSFed --times 1 --gamma 1 --gpu 6 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm FedAvg --dataset Cifar10 --num_global_iters 1000 --aggr Average --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm PerAvg --dataset Cifar10 --num_global_iters 1000 --aggr Average --times 1 --gpu 8 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --lr 0.01 --personal_lr 0.01 --beta 1 --model resnet20 --algorithm pFedMe --dataset Cifar10 --num_global_iters 1000 --aggr Average --times 1 --gpu 9 >/dev/null 2>&1 &
sleep 0.5
