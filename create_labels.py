import numpy as np
from common import DATA, load_inputs, read_jsonl, parse_answer

bank, arrays = load_inputs("train")
references = read_jsonl(DATA / "gsm8k_train.jsonl")

labels = []

for row, reference in zip(bank, references):
    assert row["id"] == reference["id"]
    assert row["question"] == reference["question"]

    gold_answer = parse_answer(reference["answer"])

    row_labels = []
    for candidate in row["candidates"]:
        candidate_answer = parse_answer(candidate["text"])
        correct = (
            candidate_answer is not None
            and candidate_answer == gold_answer
        )
        row_labels.append(int(correct))

    labels.append(row_labels)

Y = np.asarray(labels, dtype=np.int64)

output_path = DATA / "gsm8k_train_labels.npy"
np.save(output_path, Y, allow_pickle=False)

print(arrays["embeddings"].shape)  # (4755, 4, 1024)
print(Y.shape)                     # (4755, 4)
print(f"Labels saved to: {output_path}")
