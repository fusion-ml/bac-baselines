python run.py -m name=pets_lava_path alg=pets env=lava_path seed="range(5)" num_trials=250
python run.py -m name=sac_lava_path alg=sac env=lava_path seed="range(5)" num_epochs=500
python run.py -m name=td3_lava_path alg=td3 env=lava_path seed="range(5)" num_epochs=500
python run.py -m name=ppo_lava_path alg=ppo env=lava_path seed="range(5)" num_epochs=1000
