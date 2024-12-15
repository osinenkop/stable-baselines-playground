declare -a relax_prob_arr=(0.4 0.2 0.1)

for i in $(seq 105 156)
do
    for relax_prob in ${relax_prob_arr[@]}
    do
    PYTHONFAULTHANDLER=1 python ppo_vispendulum_eval_calf_wrapper.py \
        --calf-init-relax $relax_prob \
        --fallback-checkpoint "./artifacts/checkpoints/ppo_vispendulum_98304_steps.zip" \
        --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_49152_steps.zip" \
        --eval-name "fallback_25_agent_50_${relax_prob}" \
        --log --console --seed $i
    done

PYTHONFAULTHANDLER=1 python ppo_vispendulum.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_98304_steps.zip" \
    --eval-name "agent_50" \
    --log --console --seed $i

PYTHONFAULTHANDLER=1 python ppo_vispendulum.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_49152_steps.zip" \
    --eval-name "agent_25" \
    --log --console --seed $i
done
