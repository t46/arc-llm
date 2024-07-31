
def extract_output(answer, tag):
    return answer.split(f"<{tag}>")[1].split(f"</{tag}>")[0]