# Eval last checkpoint of pure PPO
python run/pendulum_ppo.py --notrain --loadstep 500000 --console

# Eval checkpoint at step 200k of pure PPO
python run/pendulum_ppo.py --notrain --loadstep 200000 --console

# Eval last checkpoint of pure PPO + CALF wrapper
python run/pendulum_ppo_load_only.py --notrain --loadstep 500000 --console

# Eval checkpoint at step 200k of pure PPO + CALF wrapper
python run/pendulum_ppo_load_only.py --notrain --loadstep 200000 --console

# Get data of EnergyBasedController as a reference
python run/pendulum_controller.py --console
