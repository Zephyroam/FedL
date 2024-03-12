nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5

nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5

nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5

nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 >/dev/null 2>&1 &
sleep 0.5

# python get_plot.py --dataset Synthetic  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --numusers 10 --gamma 1
# python get_plot.py --dataset Synthetic  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --model dnn --numusers 10 --gamma 1
# python get_plot.py --dataset Mnist  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --local_epoch 20 --gamma 1 --model dnn
# python get_plot.py --dataset Mnist  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --local_epoch 20 --gamma 1