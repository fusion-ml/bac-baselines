# python bayes_opt.py -m name=bayes_opt_weird_gain env=weird_gain seed="range(5)" hydra/launcher=joblib
# python bayes_opt.py -m name=bayes_opt_weirder_gain env=weirder_gain seed="range(5)" hydra/launcher=joblib
python bayes_opt.py -m name=bayes_opt_beta_tracking_fixed env=beta_tracking_fixed seed="range(5)" hydra/launcher=joblib
# python bayes_opt.py name=bayes_opt_beta_tracking env=beta_tracking 
# python bayes_opt.py -m name=bayes_opt_short_lava env=short_lava_path seed="range(5)" hydra/launcher=joblib
