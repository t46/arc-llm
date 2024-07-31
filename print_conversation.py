from pathlib import Path
from src.conversation import Conversation

if __name__ == '__main__':
    conv = Conversation()
    filepaths = Path('./result').glob('*.json')
    for path in filepaths:
        conv.load(path)
        conv.print_history()
