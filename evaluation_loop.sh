for i in $(seq 10 40)
do
    # Eval last checkpoint of pure PPO
    python pendulum_ppo.py --notrain --seed $i --console --log

    # Eval checkpoint at step 200k of pure PPO
    python pendulum_ppo.py --notrain --loadstep 250000 --seed $i --console --log

    # Eval last checkpoint of pure PPO + CALF wrapper
    python pendulum_ppo_load_only.py --notrain  --seed $i --console --log

    # Eval checkpoint at step 200k of pure PPO + CALF wrapper
    python pendulum_ppo_load_only.py --notrain --loadstep 250000 --seed $i --console --log

    # Get data of EnergyBasedController as a reference
    python pendulum_controller.py --seed $i --console --log
done
