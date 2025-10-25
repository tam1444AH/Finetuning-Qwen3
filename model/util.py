# model/dataset/util.py
import functools, math
import numpy as np
from transformers import TrainerCallback
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import List, Tuple, Optional


FIM_PREFIX_TOK = "<|fim_prefix|>"
FIM_MIDDLE_TOK = "<|fim_middle|>"
FIM_SUFFIX_TOK = "<|fim_suffix|>"
FIM_PAD_TOK    = "<|fim_pad|>"

@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    # 1) Try to resolve ids by name
    ids = {
        "prefix": tokenizer.convert_tokens_to_ids(FIM_PREFIX_TOK),
        "middle": tokenizer.convert_tokens_to_ids(FIM_MIDDLE_TOK),
        "suffix": tokenizer.convert_tokens_to_ids(FIM_SUFFIX_TOK),
        "pad":    tokenizer.convert_tokens_to_ids(FIM_PAD_TOK),
    }

    # 2) If any are missing, attempt to register them as *special* tokens (harmless if they already exist)
    missing = [tok for tok, tid in zip(
        [FIM_PREFIX_TOK, FIM_MIDDLE_TOK, FIM_SUFFIX_TOK, FIM_PAD_TOK],
        [ids["prefix"],  ids["middle"],  ids["suffix"],  ids["pad"]]
    ) if tid is None or tid == tokenizer.unk_token_id]

    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        # refresh ids after add
        ids = {
            "prefix": tokenizer.convert_tokens_to_ids(FIM_PREFIX_TOK),
            "middle": tokenizer.convert_tokens_to_ids(FIM_MIDDLE_TOK),
            "suffix": tokenizer.convert_tokens_to_ids(FIM_SUFFIX_TOK),
            "pad":    tokenizer.convert_tokens_to_ids(FIM_PAD_TOK),
        }

    return ids["suffix"], ids["prefix"], ids["middle"], ids["pad"]

def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400, batch_size=64):
    """
    Estimate average chars per token using token counts (works for slow & fast tokenizers).
    """
    n = min(nb_examples, len(dataset))
    if n == 0:
        return float("nan")

    total_chars = 0
    total_tokens = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        texts = [dataset[i][data_column] for i in range(start, end)]
        total_chars += sum(len(t) for t in texts)

        enc = tokenizer(
            texts,
            add_special_tokens=False,   # don't inflate with BOS/EOS/etc.
            truncation=False,           # we want true token lengths
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # enc["input_ids"] is a list of lists
        total_tokens += sum(len(ids) for ids in enc["input_ids"])

    return total_chars / max(total_tokens, 1)

def split_by_filetype(
        dataset_path: str,
    seed: int = 42,
    test_size: float = 0.10,
    filetypes: Optional[List[str]] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Load a JSON/JSONL dataset and return (train_ds, eval_ds) using the logic:
      - select rows where x["set"] == "unsupervised"
      - split each requested filetype independently with the same seed/test_size
      - concatenate the per-type train splits into train_ds
      - concatenate the per-type test splits into eval_ds
    """
    if filetypes is None:
        filetypes = ["cry", "saw", "txt"]

    # Matches your pattern: {"data": DATASET} then index ["data"]
    dataset = load_dataset("json", data_files={"data": dataset_path})["data"]

    # Keep only the unsupervised subset
    unsupervised = dataset.filter(lambda x: x.get("set") == "unsupervised")

    train_parts, eval_parts = [], []
    for ft in filetypes:
        subset = unsupervised.filter(lambda x, f=ft: x.get("filetype") == f)
        if len(subset) == 0:
            # If a type is absent, skip it gracefully
            continue
        split = subset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        train_parts.append(split["train"])
        eval_parts.append(split["test"])

    if not train_parts or not eval_parts:
        raise ValueError(
            "No data found for the specified filetypes under set=='unsupervised'. "
            "Check your dataset 'set' and 'filetype' fields."
        )

    train_ds = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    eval_ds  = concatenate_datasets(eval_parts)  if len(eval_parts)  > 1 else eval_parts[0]
    return train_ds, eval_ds

## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng


class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import math, wandb
        if metrics and "eval_loss" in metrics:
            eval_loss = float(metrics["eval_loss"])
            ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            wandb.log({
                "train/global_step": state.global_step,   # <-- drive the x-axis
                "eval/loss": eval_loss,
                "eval/perplexity": ppl,
                "train/epoch": state.epoch,
            })
            
class EpochToDatasetCallback(TrainerCallback):
    def __init__(self, dset):
        self.dset = dset
    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.dset, "set_epoch"):
            self.dset.set_epoch(state.epoch)
