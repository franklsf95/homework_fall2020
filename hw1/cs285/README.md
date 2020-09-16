# Reproducing Results

## Behavioral Cloning

Ant:

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --size 128 --num_agent_train_steps_per_iter 5000 --eval_batch_size 10000
```

Hopper:

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --size 128 --num_agent_train_steps_per_iter 5000 --eval_batch_size 10000
```

## DAgger

Ant:

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --size 128 --num_agent_train_steps_per_iter 5000 --eval_batch_size 10000
```

Hopper:

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --size 128 --num_agent_train_steps_per_iter 5000 --eval_batch_size 10000
```
