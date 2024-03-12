nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --gpu 0 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm PerAvg --dataset Synthetic --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --gpu 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm pFedMe --dataset Synthetic --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --gpu 6 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr MtoSFed --times 1 --gamma 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm PerAvg --dataset Synthetic --num_global_iters 1000 --aggr MtoSFed --times 1 --gamma 1 --gpu 0 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm pFedMe --dataset Synthetic --num_global_iters 1000 --aggr MtoSFed --times 1 --gamma 1 --gpu 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr Average --times 1 --gpu 6 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm PerAvg --dataset Synthetic --num_global_iters 1000 --aggr Average --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm pFedMe --dataset Synthetic --num_global_iters 1000 --aggr Average --times 1 --gpu 0 >/dev/null 2>&1 &
sleep 0.5
