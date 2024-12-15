declare -a relax_prob_arr=(1.0 0.75 0.5)

for i in $(seq 1 5)
do
    for relax_prob in ${relax_prob_arr[@]}
    do
    PYTHONFAULTHANDLER=1 python ppo_vispendulum_eval_calf_wrapper.py \
        --calf-init-relax $relax_prob \
        --fallback-checkpoint "./artifacts/checkpoints/ppo_vispendulum_65536_steps.zip" \
        --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_32768_steps.zip" \
        --eval-name "fallback_25_agent_50_${relax_prob}" \
        --log --console --seed $i
    done

PYTHONFAULTHANDLER=1 python ppo_vispendulum.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_65536_steps.zip" \
    --eval-name "agent_50" \
    --log --console --seed $i

PYTHONFAULTHANDLER=1 python ppo_vispendulum.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_32768_steps.zip" \
    --eval-name "agent_25" \
    --log --console --seed $i
done
