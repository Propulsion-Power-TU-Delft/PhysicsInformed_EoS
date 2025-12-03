from su2dataminer.config import Config_FGM

config = Config_FGM("OPT.cfg")
config.SetConfigName("HP_Optimization")
config.SetConcatenationFileHeader("flamelet_data_for_hp_optim")
config.SaveConfig()
