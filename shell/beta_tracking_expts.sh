# python pets.py -m name=pets_beta_tracking  env=beta_tracking seed="range(5)" num_trials=100
# python sac.py -m name=sac_beta_tracking alg=sac env=beta_tracking seed="range(5)" num_epochs=50
python td3.py -m name=td3_beta_tracking alg=td3 env=beta_tracking seed="range(5)" num_epochs=50
# python ppo.py -m name=ppo_beta_tracking alg=ppo env=beta_tracking seed="range(5)" num_epochs=50
