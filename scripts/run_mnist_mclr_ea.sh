nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg nsga2 --gpu 0 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg awga --gpu 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg moead --gpu 6 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg nsga3 --gpu 7 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm FedAvg --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg rvea --gpu 0 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm pFedMe --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg nsga2 --gpu 5 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm pFedMe --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg awga --gpu 6 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm pFedMe --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg moead --gpu 7 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm pFedMe --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg nsga3 --gpu 0 >/dev/null 2>&1 &
sleep 0.5
nohup python main.py --algorithm pFedMe --dataset Mnist --num_global_iters 1000 --aggr ParetoFed --times 1 --generation 20 --individual 10 --topk 2 --ea-alg rvea --gpu 5 >/dev/null 2>&1 &
sleep 0.5
