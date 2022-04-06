from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig



def test_net_creation():
    device = thh.get_device_id()
    config = TrainingConfig(DEVICE=device)
    net = CustomGraphNet(2, statistics=None, td=config.td)
    net.to(thh.device(config))