# python pets.py -m name=pets_plasma_tracking  env=plasma_tracking seed="range(5)" num_trials=400
python sac.py -m name=sac_plasma_tracking alg=sac env=plasma_tracking seed="range(5)" num_epochs=100
python td3.py -m name=td3_plasma_tracking alg=td3 env=plasma_tracking seed="range(5)" num_epochs=100
python ppo.py -m name=ppo_plasma_tracking alg=ppo env=plasma_tracking seed="range(5)" num_epochs=100
