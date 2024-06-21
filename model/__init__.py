from .classification.drnn import DRNN
from .classification.fasttext import FastText
from .classification.textcnn import TextCNN
from .classification.textvdcnn import TextVDCNN
from .classification.textrnn import TextRNN
from .classification.textrcnn import TextRCNN
from .classification.transformer import Transformer
from .classification.dpcnn import DPCNN
from .classification.attentive_convolution import AttentiveConvNet
from .classification.region_embedding import RegionEmbedding
from .classification.hmcn import HMCN

AVAILABLE_MODELS = {
    "DRNN": DRNN,
    "FastText": FastText,
    "TextCNN": TextCNN,
    "TextVDCNN": TextVDCNN,
    "TextRNN": TextRNN,
    "TextRCNN": TextRCNN,
    "Transformer": Transformer,
    "DPCNN": DPCNN,
    "AttentiveConvNet": AttentiveConvNet,
    "RegionEmbedding": RegionEmbedding,
    "HMCN": HMCN
}