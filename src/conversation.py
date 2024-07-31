import json

class Coversation:
    def __init__(self, save_dir):
        self.conversation = [
            {"role": "system", "content": "You are an autonomous task solver."}
        ]
        self.roles = ['system']
        self.save_dir = save_dir

    def add_user(self, txt):
        self.conversation.append({"role": "user", "content": txt})
        self.roles.append("user")

    def add_assistant(self, txt):
        self.conversation.append({"role": "assistant", "content": txt})
        self.roles.append("assistant")

    @property
    def history(self):
        return self.conversation
    
    def _print(self, conv):
        print("=" * 20)
        print(f"[{conv['role']}]")
        print("=" * 20)
        print(conv["content"])
        print('-*' * 40)

    def print(self):
        conv = self.conversation[-1]
        self._print(conv)

    def print_history(self):
        for conv in self.conversation:
            self._print(conv)
    
    def save_conversation(self):
        files = self.save_dir.glob("*.json")
        path = self.save_dir / f'{str(len(list(files))).zfill(3)}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.conversation, f, indent=4)