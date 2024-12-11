declare -a relax_prob_arr=(0.9 0.5 0.25)
declare -a decay_rate_arr=(0.01 0.005 0.001)

for i in $(seq 11 40)
do
        for relax_prob in ${relax_prob_arr[@]}
        do
                for decay_rate in ${decay_rate_arr[@]}
                do
                # PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                #         --calf-init-relax $relax_prob \
                #         --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
                #         --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
                #         --eval-name "fallback_50_agent_50_${relax_prob}"  \
                #         --log --console --seed $i 

                # PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                #         --calf-init-relax $relax_prob \
                #         --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                #         --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                #         --eval-name "fallback_25_agent_25_${relax_prob}" \
                #         --log --console --seed $i

                PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
                        --calf-init-relax $relax_prob \
                        --calf-decay-rate $decay_rate \
                        --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
                        --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
                        --eval-name "fallback_25_agent_50_${relax_prob}" \
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
        done
done