for i in $(seq 11 40)
do
        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                --fallback-checkpoint "backups/2024-12-09 101010/ppo_visual_pendulum.zip" \
                --eval-checkpoint "backups/2024-12-09 101010/ppo_visual_pendulum.zip" \
                --eval-name "fallback_100_agent_100" \
                --log --console --seed $i

        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                --fallback-checkpoint "backups/2024-12-09 101010/ppo_visual_pendulum.zip" \
                --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                --eval-name "fallback_100_agent_25" \
                --log --console --seed $i

        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
                --eval-checkpoint "backups/2024-12-09 101010/ppo_visual_pendulum.zip" \
                --eval-name "well_trained" \
                --log --console --seed $i

        PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
                --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                --eval-name "under_trained_25" \
                --log --console --seed $i

done