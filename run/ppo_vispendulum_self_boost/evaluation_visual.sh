PYTHONFAULTHANDLER=1 python ppo_vispendulum_eval_calf_wrapper.py \
    --calf-fallback-checkpoint "./artifacts/checkpoints/ppo_vispendulum_32768_steps.zip" \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_65536_steps.zip" \
    --eval-name "fallback_25_agent_50" \
    --log --console --seed 22

PYTHONFAULTHANDLER=1 python ppo_vispendulum.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_65536_steps.zip" \
    --eval-name "agent_50" \
    --log --console --seed 22

PYTHONFAULTHANDLER=1 python ppo_vispendulum.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_vispendulum_32768_steps.zip" \
    --eval-name "agent_25" \
    --log --console --seed 22
