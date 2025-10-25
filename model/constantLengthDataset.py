# model/dataset/constantLengthDataset.py
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import random
import torch 
import numpy as np
from model.dataset.util import get_fim_token_ids, permute

# Create an Iterable dataset that returns constant-length chunks of tokens from a stream of text files.

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
        already_tokenized=False,
        overlap_ratio=0.0,
        shuffle=False,
        epoch=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed
        self.already_tokenized = already_tokenized
        self.overlap_ratio = overlap_ratio
        self.shuffle = shuffle
        self.epoch = epoch
        (   self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,) = get_fim_token_ids(self.tokenizer)
        if any(tid is None for tid in (self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id)):
            if self.fim_rate > 0:
                print("[INFO] FIM tokens not available; disabling FIM.")
            self.fim_rate = 0.0
        self._length = self._estimate_length()
        

    def _estimate_length(self):
        """Estimate (and optionally measure) dataset length accurately once."""
        try:
            total_examples = 0
            total_tokens = 0
    
            iterator = iter(self.dataset)
            buffer, buffer_len = [], 0
    
            # Read through dataset once to measure token count
            for item in iterator:
                content = item[self.content_field]
                buffer.append(content)
                buffer_len += len(content)
    
            # Tokenize the whole dataset once for true token length
            if self.already_tokenized:
                all_token_ids = [tok for sub in buffer for tok in sub]
            else:
                tokenized = self.tokenizer(buffer, truncation=False)["input_ids"]
                all_token_ids = [tid for seq in tokenized for tid in seq]
    
            total_tokens = len(all_token_ids)
            stride = int(self.seq_length * (1 - self.overlap_ratio))
            stride = max(1, stride)
            total_examples = max(1, (total_tokens + stride - 1) // stride)
    
            print(f"[INFO] Empirical dataset length = {total_examples} sequences (~{total_tokens} tokens)")
            return total_examples
        except Exception as e:
            print(f"[WARN] Length estimation failed: {e}")
            return 1
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        
        stride = int(self.seq_length * (1 - self.overlap_ratio))
        stride = max(1, stride)
        
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            if self.already_tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
                
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            self.pad_tok_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
            examples = []
            for i in range(0, len(all_token_ids), stride):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) < self.seq_length:
                    input_ids = input_ids + [self.pad_tok_id] * (self.seq_length - len(input_ids))
                examples.append(input_ids)
                    
            if self.shuffle:
                # make a per-worker, per-epoch RNG for determinism
                worker_info = torch.utils.data.get_worker_info()
                wid = worker_info.id if worker_info is not None else 0
                rng = random.Random(self.seed + 1000 * self.epoch + wid)
                rng.shuffle(examples)

            for example in examples:
                inp = torch.LongTensor(example)
                labels = inp.clone()
                labels[labels == self.pad_tok_id] = -100
                yield {"input_ids": inp, "labels": labels}

    def __len__(self):
        return self._length
