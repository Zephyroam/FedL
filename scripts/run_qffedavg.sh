nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5

nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5

nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 7 >/dev/null 2>&1 &
sleep 0.5

nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 >/dev/null 2>&1 &
sleep 0.5

python get_plot.py --dataset Synthetic  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --numusers 10 --gamma 1
python get_plot.py --dataset Synthetic  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --model dnn --numusers 10 --gamma 1
python get_plot.py --dataset Mnist  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --local_epoch 20 --gamma 1 --model dnn
python get_plot.py --dataset Mnist  --num_global_iters 1000 --algorithms_list FedAvg --generation 20 --individual 10 --aggr ParetoFed qFFedAvg Average MtoSFed --local_epoch 20 --gamma 1

nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.01 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.1 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 10 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 100 >/dev/null 2>&1 &

nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.01 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.1 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 10 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 100 >/dev/null 2>&1 &

nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.01 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.1 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 10 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 100 >/dev/null 2>&1 &

nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.01 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 0.1 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 10 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 100 >/dev/null 2>&1 &


nohup python main.py --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --personal_lr 0.01 --numusers 10 --algorithm FedAvg --dataset Synthetic --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --model dnn --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr qFFedAvg --times 1 --gpu 0 --q 5 >/dev/null 2>&1 &


python get_table.py --dataset Mnist --models dnn mclr --num_global_iters 1000 --algorithms_list FedAvg --aggrs qFFedAvg --length 100 --end 800 --q 1
python get_table.py --dataset Mnist --models dnn mclr --num_global_iters 1000 --algorithms_list FedAvg --aggrs qFFedAvg --length 100 --end 800 --q 0.01 0.1 5 10

python get_table.py --dataset Synthetic --numusers 10 --models dnn mclr --num_global_iters 1000 --algorithms_list FedAvg --aggrs qFFedAvg --length 100 --end 800
python get_table.py --dataset Synthetic --numusers 10 --models dnn mclr --num_global_iters 1000 --algorithms_list FedAvg --aggrs qFFedAvg --length 100 --end 800 --q 0.01 0.1 5 10