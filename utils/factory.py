from methods.palora import PaLoRA
from methods.sprompt_coda import SPrompts_coda
from methods.sprompt_l2p import SPrompts_l2p
from methods.sprompt_dual import SPrompts_dual
from methods.lorasub_drs import LoRAsub_DRS
from methods.bilora import BiLoRA
from methods.inflora import InfLoRA
from methods.sequence_lora import LoRA
from methods.pretrain import Pretrain
def get_model(model_name, args):
    name = model_name.lower()
    options = {'palora': PaLoRA,
               'inflora': InfLoRA,
               'bilora': BiLoRA,
               'lorasub_drs': LoRAsub_DRS,
               'sprompts_coda': SPrompts_coda,
               'sprompts_l2p': SPrompts_l2p,
               'sprompts_dual': SPrompts_dual,
               'sequence': LoRA,
               'pretrain': Pretrain,
               }
    return options[name](args)

