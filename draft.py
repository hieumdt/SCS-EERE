from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector
import torch
from transformers import RobertaConfig

RobertaConfig._to_dict_new = RobertaConfig.to_dict


selector = torch.load('MATRES_model/selector.pth')
torch.save(selector.state_dict(), 'MATRES_model/selector.pt')
predictor = torch.load('MATRES_model/predictor.pth')
torch.save(predictor.state_dict(), 'MATRES_model/predictor.pt')


