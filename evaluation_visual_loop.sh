for i in $(seq 11 40)
do
        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
                --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
                --eval-name "fallback_50_agent_50" \
                --log --console --seed $i

        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                --eval-name "fallback_25_agent_25" \
                --log --console --seed $i

        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
                --eval-name "fallback_25_agent_50" \
                --log --console --seed $i

        # PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
        #         --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
        #         --eval-name "agent_50" \
        #         --log --console --seed $i

        # PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
        #         --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
        #         --eval-name "agent_25" \
        #         --log --console --seed $i

done