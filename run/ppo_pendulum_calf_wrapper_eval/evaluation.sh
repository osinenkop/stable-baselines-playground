# Eval last checkpoint of pure PPO
python run/ppo_pendulum_calf_wrapper_eval/pendulum_ppo.py --notrain --loadstep 500000 --console

# Eval checkpoint at step 200k of pure PPO
python run/ppo_pendulum_calf_wrapper_eval/pendulum_ppo.py --notrain --loadstep 200000 --console

# Eval last checkpoint of pure PPO + CALF wrapper
python run/ppo_pendulum_calf_wrapper_eval/pendulum_ppo_load_only.py --notrain --loadstep 500000 --console

# Eval checkpoint at step 200k of pure PPO + CALF wrapper
python run/ppo_pendulum_calf_wrapper_eval/pendulum_ppo_load_only.py --notrain --loadstep 200000 --console

# Get data of EnergyBasedController as a reference
python run/ppo_pendulum_calf_wrapper_eval/pendulum_controller.py --console
