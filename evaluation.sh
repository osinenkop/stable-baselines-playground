# Eval last checkpoint of pure PPO
python pendulum_ppo.py --notrain

# Eval checkpoint at step 200k of pure PPO
python pendulum_ppo.py --notrain --loadstep 200000

# Eval last checkpoint of pure PPO + CALF wrapper
python pendulum_ppo_load_only.py --notrain 

# Eval checkpoint at step 200k of pure PPO + CALF wrapper
python pendulum_ppo_load_only.py --notrain --loadstep 200000

# Get data of EnergyBasedController as a reference
python pendulum_controller.py --notrain
