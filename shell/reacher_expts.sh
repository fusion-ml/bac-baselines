# python pets.py -m name=pets_reacher env=reacher seed="range(5)" num_trials=50
# python sac.py -m name=sac_reacher alg=sac env=reacher seed="range(5)" num_epochs=100

python td3.py -m name=td3_reacher alg=td3 env=reacher seed="range(5)" num_epochs=100

# python ppo.py -m name=ppo_reacher alg=ppo env=reacher seed="range(5)" num_epochs=100

