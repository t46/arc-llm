import sys
from pathlib import Path
from src.conversation import Conversation

if __name__ == '__main__':
    read_dir = Path('./result/raw/exp2')
    save_dir = Path('./result/print/exp2')
    save_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(Path(read_dir).glob('*.json'))
    conv = Conversation()
    for path in filepaths:
        print(path)
        conv.load(path)

        # # for avoid the long DSL prompt
        # conv.conversation = conv.conversation[2:]

        savepath = save_dir / (path.stem + '.txt')
        with open(savepath, 'w') as f:
            sys.stdout = f
            conv.print_history()
            sys.stdout = sys.__stdout__
