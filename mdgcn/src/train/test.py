import sys
import os
sys.path.append(os.path.abspath('../..'))
from src.train.train import Config
from src.train.train import TrainTool as TestTool
from src.train.outstream import tprint


cfg = Config(dataset="kickstarter", weights_name="")
cfg.test_mode = True
test_tool = TestTool(cfg)
tprint("Start...")
res = test_tool.eval(test_tool.test_dataloader)
tprint("End")
tprint(res, display_time=True)







