import pickle
from pathlib import Path
import re

__version__ = "0.1.0"

BASEDIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASEDIR}/trained_pipeline-{__version__}.pkl","rb") as f:
    model = pickle.load(f)

classes = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']


def predict_pipeline(text):
    pred = model.predict([text])
    return pred

