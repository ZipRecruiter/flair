import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import flair
from flair.data import Dictionary, Sentence
from flair.nn import Classifier

logger = logging.getLogger(__name__)


def _get_separator_id(tokenizer: AutoTokenizer, sep_token: str) -> int:
    """Return the numeric id for a separator token.

    Args:
        tokenizer: Tokenizer for the model.
        sep_token: The separator token string.

    Returns:
        The numeric ID of the separator token.
    """
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    if sep_id is None or sep_id == tokenizer.unk_token_id:
        ids = tokenizer.encode(sep_token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Separator '{sep_token}' tokenizes into multiple IDs ({ids})."
                " Please choose a separator that corresponds to a single token."
            )
        sep_id = ids[0]
    return sep_id


@dataclass
class TrieNode:
    """Trie for fast prefix-based filtering of label token sequences."""

    children: dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False

    def insert(self, token_ids: list[int]) -> None:
        """Insert a sequence of token IDs into this trie."""
        node = self
        for tok in token_ids:
            node = node.children.setdefault(tok, TrieNode())
        node.is_end = True

    def walk(self, token_ids: list[int]) -> Optional["TrieNode"]:
        """Traverse the trie based on the given token IDs.

        Args:
            token_ids: List of token IDs to traverse.

        Returns:
            The TrieNode at the end of the traversal, or None if the path does not exist.
        """
        node = self
        for t in token_ids:
            node = node.children.get(t)
            if node is None:
                return None
        return node


def build_label_trie(tokenizer: AutoTokenizer, vocab: list[str]) -> TrieNode:
    """Builds a TrieNode from a list of label strings.

    Args:
        tokenizer: Tokenizer for the model.
        vocab: List of label strings to insert into the trie.

    Returns:
        A TrieNode containing the label vocabulary.
    """
    trie = TrieNode()
    for label in vocab:
        label = label.strip()
        if not label:
            logger.warning("Empty label found in vocabulary. Skipping.")
            continue
        token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        if token_ids:
            trie.insert(token_ids)
    return trie


def make_prefix_allowed_tokens_fn(
    trie: TrieNode, tokenizer: AutoTokenizer, prompt_lengths: list[int], sep_id: int
) -> Callable[[int, torch.LongTensor], list[int]]:
    """Factory that returns a prefix_allowed_tokens_fn for generate().

    Args:
        trie: TrieNode containing the label vocabulary.
        tokenizer: Tokenizer for the model.
        prompt_lengths: List of prompt lengths for each batch item. This is used to determine the start of the label segment.
        sep_id: ID of the separator token.

    Returns:
        A `prefix_allowed_tokens_fn` function that filters allowed tokens based on the current prefix.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def prefix_fn(batch_id: int, full_ids: torch.LongTensor) -> list[int]:
        """This function is used by the tranformer library as a prefix_allowed_tokens_fn.

        It is called during generation to filter the allowed tokens based on the current prefix.
        This ensures that only valid continuations of labels are generated.

        Example Output Format:
        <prompt_tokens> label_1 <sep> label_2 <sep> ... <sep> label_n <eos>
        """
        # remove padding tokens
        ids = full_ids.view(-1).tolist()
        while ids and ids[0] == pad_id:
            ids.pop(0)

        # remove prompt tokens
        ids = ids[prompt_lengths[batch_id] :]

        # no label tokens have been generated yet; all starting tokens are valid
        if not ids:
            return list(trie.children.keys())

        # find the start of the current label segment (tokens since the last separator)
        separator_indices = [i for i, x in enumerate(ids) if x == sep_id]
        last_sep = separator_indices[-1] if separator_indices else -1
        label_segment = ids[last_sep + 1 :]

        # find all valid continuations for this label segment
        node = trie.walk(label_segment)
        if node is None:
            # no valid continuations; return only EOS
            return [eos_id]
        allowed = list(node.children.keys())
        if node.is_end:
            # if this segment is a valid label, allow EOS and the separator token
            allowed += [eos_id, sep_id]

        return allowed if allowed else [eos_id]

    return prefix_fn


class GenerativeClassifier(Classifier):
    """Generative classifier leveraging Causal LMs for text classification.

    This model fine-tunes a causal language model (e.g. Qwen, Phi, GPT-2) to generate
    label strings based on the input text. It supports two modes:
    1. Constrained Decoding: Uses a prefix Trie built from the `label_dictionary`
       to force the model to only generate valid, predefined labels separated by `separator`.
    2. Open-Set Generation: Allows the model to generate any text after the prompt.
       The generated text is then split by `separator` to determine the predicted labels.
    """

    def __init__(
        self,
        causal_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        label_dictionary: Dictionary,
        label_type: str,
        prompt_template: Union[str, Callable[[str], str]] = "{text}",
        mask_input: bool = True,
        separator: str = ",",
        use_constrained_decoding: bool = True,
        generation_kwargs: Optional[dict[str, Any]] = None,
        label_rank_map: Optional[dict[str, int]] = None,
        label_sort_fn: Optional[Callable[[list[str]], list[str]]] = None,
        trust_remote_code: bool = False,
        max_length: int = 1024,
    ) -> None:
        """Initializes a GenerativeClassifier.

        Args:
            causal_model: The pre-trained causal language model.
            tokenizer: The tokenizer associated with the causal model.
            label_dictionary: A Dictionary containing all predictable labels from the corpus
            label_type: The label type which is going to be predicted
            prompt_template: String template ('{text}' placeholder required) or callable to format input text into the LM prompt. Defaults to "{text}".
            mask_input: If True, masks input prompt tokens during loss calculation. Defaults to True.
            separator: String separating multiple labels in generated/gold sequences. Must be a single token. Defaults to ",".
            use_constrained_decoding: If True, use Trie-based constrained decoding during prediction; otherwise, use unconstrained generation. Defaults to True.
            generation_kwargs: Arguments passed to HuggingFace `generate()` during prediction. `max_new_tokens` defaults to 64. Defaults to {}.
            label_rank_map: Map from label string to rank for consistent sorting (lower rank first). Used if `label_sort_fn` is None. Defaults to None.
            label_sort_fn: Function to sort gold labels for target string creation. If None, uses `label_rank_map` or alphabetical order. Defaults to None.
            trust_remote_code: If True, allows loading models with remote code. Defaults to False.
            max_length: Maximum sequence length (prompt + generated tokens) for truncation. Defaults to 1024.
        """
        super().__init__()

        if isinstance(prompt_template, str) and "{text}" not in prompt_template:
            raise ValueError("String `prompt_template` must contain the literal placeholder '{text}'.")
        self.prompt_template: Union[str, Callable[[str], str]] = prompt_template

        if trust_remote_code:
            logger.warning(
                "Initializing GenerativeClassifier with `trust_remote_code=True`. Ensure you trust the source of the causal_model."
            )
        self._trust_remote_code = trust_remote_code

        conflicting_labels = [label for label in label_dictionary.get_items() if separator in label]
        if conflicting_labels:
            logger.warning(
                f"The following labels contain the separator '{separator}': {conflicting_labels}. "
                "This may lead to incorrect behavior during decoding. "
                "Consider choosing a different separator or modifying the affected labels."
            )

        self.causal_model = causal_model.to(flair.device)
        self.label_dictionary = label_dictionary
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # decoder-only models require left padding for batched generation

        self._label_type = label_type
        self.separator = separator
        self._sep_id = _get_separator_id(tokenizer, separator)

        self.use_constrained_decoding = use_constrained_decoding
        self._trie: Optional[TrieNode] = None
        if self.use_constrained_decoding:
            logger.info("Building label Trie for constrained decoding...")
            self._trie = build_label_trie(tokenizer, label_dictionary.get_items())
            logger.info("Label Trie built.")
        else:
            logger.info("`use_constrained_decoding` is False. Prediction will use unconstrained generation.")

        self.mask_input = mask_input
        self.max_length = max_length

        self.generation_kwargs = generation_kwargs or {}
        self.generation_kwargs.setdefault("max_new_tokens", 128)
        self.generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        self.generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        self._label_rank_map = label_rank_map or {}
        self._sort_fn: Callable[[list[str]], list[str]] = (
            label_sort_fn or (lambda labels: sorted(labels, key=lambda label: self._label_rank_map.get(label, float("inf"))))
            if self._label_rank_map
            else sorted
        )

        self.train()

    @property
    def label_type(self) -> str:
        return self._label_type

    def _format_prompt(self, text: str) -> str:
        """Return full string fed into the LM."""
        return self.prompt_template(text) if callable(self.prompt_template) else self.prompt_template.format(text=text)

    def _gold_string(self, s: Sentence) -> str:
        """Creates the target string of sorted labels."""
        labels = [label.value for label in s.get_labels(self.label_type)]
        sorted_labels = self._sort_fn(labels)
        return f"{self.separator} ".join(sorted_labels)

    def forward_loss(self, sentences: list[Sentence]) -> tuple[torch.Tensor, int]:
        if not sentences:
            return torch.zeros(1, device=flair.device, requires_grad=True), 0

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        seq_tensors, label_tensors = [], []
        processed_count = 0

        for sentence in sentences:
            prompt = self._format_prompt(sentence.text)
            gold_str = self._gold_string(sentence)

            if not gold_str:
                continue

            gold_string_ids = self.tokenizer.encode(gold_str, add_special_tokens=False)

            max_prompt_len = max(0, self.max_length - len(gold_string_ids) - 1)  # reserve room for gold + EOS
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_len if max_prompt_len else 1,
            )

            seq = prompt_ids + gold_string_ids + [eos_id]
            # mask input tokens with -100 for loss calculation if self.mask_input is True
            label = ([-100] * len(prompt_ids) + gold_string_ids + [eos_id]) if self.mask_input else seq

            if len(seq) > self.max_length:
                logger.error(
                    f"Sequence length ({len(seq)}) exceeds max_length ({self.max_length}) after truncation. Skipping."
                )
                continue

            seq_tensors.append(torch.tensor(seq, device=flair.device, dtype=torch.long))
            label_tensors.append(torch.tensor(label, device=flair.device, dtype=torch.long))
            processed_count += 1

        if not processed_count:
            return torch.zeros(1, device=flair.device, requires_grad=True), 0

        inputs = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        attn_mask = (inputs != pad_id).long()

        outputs = self.causal_model(
            input_ids=inputs,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss

        if self.training and not loss.requires_grad:
            loss = loss.clone().requires_grad_(True)

        return loss, processed_count

    def predict(
        self,
        sentences: Union[Sentence, list[Sentence]],
        mini_batch_size: int = 8,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        generation_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[tuple[torch.Tensor, int]]:
        if isinstance(sentences, Sentence):
            sentences = [sentences]
        if not sentences:
            return (torch.zeros(1, device=flair.device), 0) if return_loss else None

        self.causal_model.eval()
        is_training = self.training
        self.training = False

        final_generation_kwargs = self.generation_kwargs.copy()
        if generation_kwargs:
            final_generation_kwargs.update(generation_kwargs)

        label_type = label_name or self.label_type
        all_losses = []
        total_count = 0

        with torch.no_grad():
            for i in range(0, len(sentences), mini_batch_size):
                batch = sentences[i : i + mini_batch_size]
                prompts = [self._format_prompt(s.text) for s in batch]

                # calculate max prompt length to reserve space for max_new_tokens when generating
                prompt_max_len = max(1, self.max_length - (final_generation_kwargs.get("max_new_tokens") or 0))
                tok = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=prompt_max_len,
                    return_tensors="pt",
                ).to(flair.device)

                prefix_fn = None
                if self.use_constrained_decoding:
                    if not self._trie:
                        raise RuntimeError("Constrained decoding enabled, but Trie was not built.")

                    # get the prompt lengths for the prefix function.
                    prompt_lengths = tok["attention_mask"].sum(dim=1).tolist()
                    prefix_fn = make_prefix_allowed_tokens_fn(self._trie, self.tokenizer, prompt_lengths, self._sep_id)

                gen_ids = self.causal_model.generate(
                    **tok,
                    **final_generation_kwargs,
                    prefix_allowed_tokens_fn=prefix_fn,
                )

                gen_only = gen_ids[:, tok["input_ids"].shape[1] :]  # remove prompt tokens
                outputs = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

                for sentence, out_str in zip(batch, outputs):
                    sentence.remove_labels(label_type)
                    generated_items = {s.strip() for s in out_str.split(self.separator) if s.strip()}
                    for item in generated_items:
                        if not self.use_constrained_decoding or self.label_dictionary.has_item(item):
                            sentence.add_label(label_type, item)

                if return_loss:
                    loss, count = self.forward_loss(batch)
                    if count > 0:
                        all_losses.append(loss * count)
                        total_count += count

        self.training = is_training
        self.causal_model.train(is_training)

        if return_loss:
            if total_count > 0:
                final_loss = sum(all_losses) / total_count
                return final_loss, total_count
            else:
                return torch.zeros(1, device=flair.device), 0
        else:
            return None

    def _get_state_dict(self) -> dict[str, Any]:
        savable_prompt_template = None
        if isinstance(self.prompt_template, str):
            savable_prompt_template = self.prompt_template

        return {
            **super()._get_state_dict(),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "lm_model_name": self.causal_model.config._name_or_path,
            "max_length": self.max_length,
            "prompt_template": savable_prompt_template,
            "mask_input": self.mask_input,
            "separator": self.separator,
            "use_constrained_decoding": self.use_constrained_decoding,
            "generation_kwargs": self.generation_kwargs,
            "label_rank_map": self._label_rank_map,
            "lm_state_dict": self.causal_model.state_dict(),
            "trust_remote_code": getattr(self, "_trust_remote_code", False),
        }

    @classmethod
    def _init_model_with_state_dict(cls, state: dict[str, Any], **kwargs) -> "GenerativeClassifier":
        prompt_template = state.get("prompt_template")
        if not prompt_template:
            logger.warning(
                "Checkpoint contained a non-string `prompt_template` (likely a callable). "
                "Such templates cannot be restored automatically; using default '{text}'. "
                "Re-assign `model.prompt_template` manually if needed."
            )
            prompt_template = "{text}"

        label_dictionary = state["label_dictionary"]
        label_type = state["label_type"]
        mask_input = state.get("mask_input", True)
        separator = state.get("separator", ",")
        use_constrained_decoding = state.get("use_constrained_decoding", True)
        generation_kwargs = state.get("generation_kwargs", {})
        generation_kwargs.setdefault("max_new_tokens", 128)
        label_rank_map = state.get("label_rank_map")
        max_length = state.get("max_length", 1024)
        trust_remote_code = state.get("trust_remote_code", False)

        lm_name = state["lm_model_name"]
        tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=trust_remote_code)

        # Load model and update vocab size
        config = AutoConfig.from_pretrained(lm_name, trust_remote_code=trust_remote_code)
        saved_vocab_size = state["lm_state_dict"]["model.embed_tokens.weight"].size(0)
        config.vocab_size = saved_vocab_size
        causal_model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        causal_model.load_state_dict(state["lm_state_dict"])

        model = cls(
            causal_model=causal_model,
            tokenizer=tokenizer,
            label_dictionary=label_dictionary,
            label_type=label_type,
            prompt_template=prompt_template,
            mask_input=mask_input,
            separator=separator,
            use_constrained_decoding=use_constrained_decoding,
            generation_kwargs=generation_kwargs,
            label_rank_map=label_rank_map,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return model
