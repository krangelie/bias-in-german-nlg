"""Create adversarial trigger."""
import collections
import heapq
import os, sys
import string
from copy import deepcopy

import hydra.utils
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    GPT2Tokenizer,
)

import src.constants as constants
import src.bias_mitigator.attacks as attacks

from src.bias_mitigator.loss_related_operations import (
    get_target_losses_and_threshold,
    interleave_losses,
    get_loss,
)
from src.bias_mitigator.prepare_target_texts import prepare_texts, strip_bias_context
from src.bias_mitigator.utils import set_device_and_seeds, adjust_params


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    global extracted_grads
    extracted_grads.append(grad_out[0])


# Returns the wordpiece embedding weight matrix.
def get_embedding_weight(language_model, vocab_size):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if (
                module.weight.shape[0] == vocab_size
            ):  # Only add a hook to wordpiece embeddings, not position embeddings.
                return module.weight.detach()


# Add hooks for embeddings.
def add_hooks(language_model, vocab_size):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if (
                module.weight.shape[0] == vocab_size
            ):  # Only add a hook to wordpiece embeddings, not position.
                module.weight.requires_grad = (
                    True  # allow update of wordpiece embedding
                )
                module.register_backward_hook(extract_grad_hook)


def get_salience_token_items(params, tokenizer, total_vocab_size):
    salience_dict = attacks.find_hard_salient_phrases(
        hydra.utils.to_absolute_path(params.neg_sample_file),
        hydra.utils.to_absolute_path(params.pos_sample_file),
        tokenizer,
        hydra.utils.to_absolute_path(params.salient_phrases_file),
        salience_threshold=params.salience_threshold,
    )
    neg_salience_token_items = [0] * total_vocab_size
    pos_salience_token_items = [0] * total_vocab_size
    for phrase in salience_dict:
        label, score = salience_dict[phrase]
        tok_ids = tokenizer.encode(phrase)
        if label == "neg":
            for tok_id in tok_ids:
                neg_salience_token_items[tok_id] += int(round(score))
        elif label == "pos":
            for tok_id in tok_ids:
                pos_salience_token_items[tok_id] += int(round(score))
        else:
            raise NotImplementedError("Label is either neg or pos.")
    print("neg_salience_token_items", neg_salience_token_items[:20])
    print("pos_salience_token_items", pos_salience_token_items[:20])
    return neg_salience_token_items, pos_salience_token_items


def get_trigger_masked_phrases(params, enc_trigger_init, tokenizer):
    # Process trigger_masked_phrases. -- ??
    trigger_masked_idxes = []
    for phrase in params.trigger_masked_phrases:
        enc_phrase = tokenizer.encode(phrase)
        enc_trigger_init_str = " ".join([str(x) for x in enc_trigger_init])
        enc_phrase_str = " ".join([str(x) for x in enc_phrase])
        if enc_phrase_str in enc_trigger_init_str:
            enc_phrase_str_char_idx = enc_trigger_init_str.index(
                enc_phrase_str
            )  # where in initial trigger is the beginning of phrase
            start_idx = enc_trigger_init_str[:enc_phrase_str_char_idx].count(
                " "
            )  # how many words before phrase
            for i in range(start_idx, start_idx + len(enc_phrase)):
                trigger_masked_idxes.append(
                    i + params.num_trigger_tokens - 1
                )  # at each index of phrase in sentence, save idx (offset by number of trigger
                # tokens)
        else:  # Try adding space before the phrase bc of tokenization.
            sp_enc_phrase = tokenizer.encode("x " + phrase)[1:]
            sp_enc_phrase_str = " ".join([str(x) for x in sp_enc_phrase])
            if sp_enc_phrase_str in enc_trigger_init_str:
                sp_enc_phrase_str_char_idx = enc_trigger_init_str.index(
                    sp_enc_phrase_str
                )
                start_idx = enc_trigger_init_str[:sp_enc_phrase_str_char_idx].count(" ")
                for i in range(start_idx, start_idx + len(sp_enc_phrase)):
                    trigger_masked_idxes.append(i + params.num_trigger_tokens - 1)
            else:
                print(
                    "Masked phrase not found",
                    enc_phrase,
                    sp_enc_phrase,
                    enc_trigger_init,
                )
                exit()
    print("trigger_masked_idxes", trigger_masked_idxes)
    return trigger_masked_idxes


def get_model_and_tokenizer(params):
    # Load GPT
    if params.model_type == constants.GPT2:
        tokenizer = AutoTokenizer.from_pretrained(params.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(params.model_name_or_path)
    elif params.model_type == constants.GPTNEO:
        model = GPTNeoForCausalLM.from_pretrained(params.model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(params.model_name_or_path)
    return model, tokenizer


def create_adversarial_tokens(params):
    global extracted_grads
    params = params.run_mode
    OmegaConf.set_struct(params, False)  # allows overriding conf
    print("Redirecting stdout to 'outputs' folder.")
    orig_stdout = sys.stdout
    f = open("trigger_search_stdout.txt", "w")
    sys.stdout = f
    params = adjust_params(params)
    device = set_device_and_seeds()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(params)
    total_vocab_size = len(tokenizer)
    model.eval()
    model.to(device)

    add_hooks(model, total_vocab_size)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(
        model, total_vocab_size
    )  # save the word embedding matrix

    enc_trigger_init = tokenizer.encode("Das " + params.trigger_init)[
        1:
    ]  # this will be "" if not otherwise specified
    trigger_init_len = len(enc_trigger_init)
    old_num_trigger_tokens = params.num_trigger_tokens
    params.num_trigger_tokens = max(trigger_init_len, params.num_trigger_tokens)

    trigger_masked_idxes = get_trigger_masked_phrases(
        params, enc_trigger_init, tokenizer
    )

    # Calculate salience scores.
    neg_salience_token_items, pos_salience_token_items = [], []
    if params.use_salience_loss:
        neg_salience_token_items, pos_salience_token_items = get_salience_token_items(
            params, tokenizer, total_vocab_size
        )

    (
        neg_demo_neg_target_texts,
        neg_demo_neu_target_texts,
        neg_demo_pos_target_texts,
        neg_names,
        pos_demo_neg_target_texts,
        pos_demo_neu_target_texts,
        pos_demo_pos_target_texts,
        pos_names,
    ) = prepare_texts(params)

    if (
        constants.BODY in params.trigger_position
        and constants.BC in params.trigger_position
    ):
        (
            neg_demo_neg_target_texts,
            neg_demo_neu_target_texts,
            neg_demo_pos_target_texts,
            pos_demo_neg_target_texts,
            pos_demo_neu_target_texts,
            pos_demo_pos_target_texts,
        ) = strip_bias_context(
            neg_demo_neg_target_texts,
            neg_demo_neu_target_texts,
            neg_demo_pos_target_texts,
            pos_demo_neg_target_texts,
            pos_demo_neu_target_texts,
            pos_demo_pos_target_texts,
        )

    print("neg demo neg target text:", neg_demo_neg_target_texts[0])
    print("pos demo pos target text:", pos_demo_pos_target_texts[0])

    if params.use_dissociation_loss:
        print("pos demo neg target text:", pos_demo_neg_target_texts[0])
        print("neg demo pos target text:", neg_demo_pos_target_texts[0])

    if params.neu_sample_file:
        print("neg demo neu target text:", neg_demo_neu_target_texts[0])
        print("pos demo neu target text:", pos_demo_neu_target_texts[0])

    (
        diff_demo_target_losses,
        diff_demo_target_threshold,
        neu_target_losses,
        neu_target_threshold,
        same_demo_target_losses,
        same_demo_target_threshold,
    ) = get_target_losses_and_threshold(
        params,
        tokenizer,
        neg_demo_neg_target_texts,
        neg_demo_neu_target_texts,
        neg_demo_pos_target_texts,
        pos_demo_neg_target_texts,
        pos_demo_neu_target_texts,
        pos_demo_pos_target_texts,
    )

    all_items = interleave_losses(
        diff_demo_target_losses,
        diff_demo_target_threshold,
        neu_target_losses,
        neu_target_threshold,
        params,
        same_demo_target_losses,
        same_demo_target_threshold,
    )

    run_trigger_search_loop(
        params,
        model,
        tokenizer,
        all_items,
        embedding_weight,
        neg_names,
        pos_names,
        neg_salience_token_items,
        pos_salience_token_items,
        old_num_trigger_tokens,
        trigger_init_len,
        trigger_masked_idxes,
    )
    sys.stdout = orig_stdout
    f.close()


def run_trigger_search_loop(
    params,
    model,
    tokenizer,
    all_items,
    embedding_weight,
    neg_names,
    pos_names,
    neg_salience_token_items,
    pos_salience_token_items,
    old_num_trigger_tokens,
    trigger_init_len,
    trigger_masked_idxes,
):
    global extracted_grads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for restart_idx in range(1):  # Different random restarts of the trigger
        print("Random restart: ", str(restart_idx))

        trigger_tokens = tokenizer.encode("Das " + params.trigger_init)[1:]
        if trigger_init_len < old_num_trigger_tokens:
            # Sample random initial trigger.
            # rand_trigger_tokens = np.random.randint(total_vocab_size, size=old_num_trigger_tokens - trigger_init_len)
            rand_trigger_tokens = [tokenizer.encode("x das")[-1]] * (
                old_num_trigger_tokens - trigger_init_len
            )
            trigger_tokens = np.concatenate(
                (trigger_tokens, rand_trigger_tokens), axis=0
            )
        if params.model_type == constants.DIALOGPT:  # Add eos after trigger.
            trigger_tokens = np.concatenate(
                (trigger_tokens, [tokenizer.eos_token_id]), axis=0
            )
        print("Random initial trigger:", tokenizer.decode(trigger_tokens))

        # Note that beam_cache, new_beam_cache, and loss_heap all have reverse sign losses.
        # best_loss and curr_best_loss have original sign losses.
        best_loss = 999999  # We want to minimize loss.
        best_trigger_tokens = deepcopy(trigger_tokens)
        beam_cache = [
            (-999999, trigger_tokens)
        ]  # Always keep beam_size full trigger candidates.
        end_iter = False
        for entire_trigger_update_idx in range(
            10  # 50
        ):  # this many updates of the entire trigger sequence
            print(
                "Updating entire trigger for the",
                str(entire_trigger_update_idx),
                "-th time",
            )

            if end_iter:
                continue

            for token_to_flip in range(params.num_trigger_tokens):
                right_counter_token_to_flip = token_to_flip

                if token_to_flip in trigger_masked_idxes:
                    print(
                        "Trigger token #",
                        str(token_to_flip),
                        str(right_counter_token_to_flip),
                    )
                    continue  # Don't modify these triggers.

                beam_cache, trigger_tokens = run_beam_search(
                    params,
                    device,
                    model,
                    tokenizer,
                    all_items,
                    beam_cache,
                    trigger_tokens,
                    embedding_weight,
                    neg_names,
                    pos_names,
                    neg_salience_token_items,
                    pos_salience_token_items,
                    right_counter_token_to_flip,
                    token_to_flip,
                )

            curr_best_loss = 999999
            for x, y in beam_cache:
                x *= -1  # Flip loss back to original sign.
                if x < curr_best_loss:
                    curr_best_loss = x
                    trigger_tokens = deepcopy(y)
            print("Loss: " + str(curr_best_loss))
            print("Trigger token IDs:", trigger_tokens)
            print("Trigger string:", tokenizer.decode(trigger_tokens) + "\n")
            if curr_best_loss < best_loss:
                best_loss = curr_best_loss
                best_trigger_tokens = deepcopy(trigger_tokens)
            elif curr_best_loss == best_loss:
                pass
            else:
                end_iter = True

        # Print final trigger.
        condition_string = (
            f"debias-{params.debias}_assoc-loss-"
            f"{params.use_original_loss}_diss-loss-{params.use_dissociation_loss}_pos-demo-{params.pos_demographic}_neg-demo-{params.neg_demographic}"
        )
        output_path = hydra.utils.to_absolute_path(params.output_path)
        dest = os.path.join(output_path, condition_string)
        os.makedirs(dest, exist_ok=True)
        # with open(os.path.join(dest, "triggers.txt"), "w") as f:
        # condition_string = condition_string.replace("_", "\n")
        # f.write(condition_string)
        print("\nFinal loss: " + str(best_loss))
        # f.write("Final loss: " + str(best_loss))
        print("Final trigger token IDs:", best_trigger_tokens)
        # f.write("\nFinal trigger token IDs: " + best_trigger_tokens)
        decoded_tokens = tokenizer.decode(best_trigger_tokens)
        print("Final trigger:", decoded_tokens)
        # f.write("\nFinal trigger: " + decoded_tokens)
        # f.write(f"{condition_string}")
        # params.trigger_dump_file = "_stdout.txt"


def run_beam_search(
    params,
    device,
    model,
    tokenizer,
    all_items,
    beam_cache,
    trigger_tokens,
    embedding_weight,
    neg_names,
    pos_names,
    neg_salience_token_items,
    pos_salience_token_items,
    right_counter_token_to_flip,
    token_to_flip,
):
    global extracted_grads
    # Beam search for each trigger_tokens in beam_cache.
    assert len(beam_cache) <= params.beam_size
    new_beam_cache = []
    for _, trigger_tokens in beam_cache:
        print(
            "Trigger token #",
            str(token_to_flip),
            str(right_counter_token_to_flip),
        )
        print(tokenizer.decode(trigger_tokens), trigger_tokens)

        model.zero_grad()
        extracted_grads = (
            []
        )  # Each element is (batch_size, sample_length, 768_embed_dim).
        loss_types = []  # Order of `add` and `sub` loss types.
        demo_types = []  # Order of `neg` or `pos` demographic types.
        for idx, (typ, demo_type, target_tokens) in enumerate(all_items):
            loss_types.append(typ)
            demo_types.append(demo_type)

            if demo_type == "neg":
                if params.neg_name_file:
                    demo_tokens = [tokenizer.encode("The " + n)[1:] for n in neg_names]
                else:
                    demo_tokens = tokenizer.encode(params.neg_demographic)
            elif demo_type == "pos":
                if params.pos_name_file:
                    demo_tokens = [tokenizer.encode("The " + n)[1:] for n in pos_names]
                else:
                    demo_tokens = tokenizer.encode(params.pos_demographic)
            else:
                raise NotImplementedError("Label is either neg or pos.")

            if params.use_salience_loss and not params.debias:
                if (demo_type == "neg" and typ == "add") or (
                    demo_type == "pos" and typ == "sub"
                ):
                    # association loss between neg demographic and neg samples (to add) or
                    # association loss between pos demographic and neg samples (to subtract).
                    salience_token_items = neg_salience_token_items
                elif (demo_type == "pos" and typ == "add") or (
                    demo_type == "neg" and typ == "sub"
                ):
                    # association loss between pos demographic and pos samples (to add) or
                    # association loss between neg demographic and pos samples (to subtract).
                    salience_token_items = pos_salience_token_items
                else:
                    raise NotImplementedError(
                        "Label and demographic pair not possible",
                        typ,
                        demo_type,
                    )
                salience_token_items_tensor = torch.tensor(
                    salience_token_items, device=device, dtype=torch.long
                )
            else:
                salience_token_items_tensor = None

            loss, _ = get_loss(
                model,
                params.batch_size,
                trigger_tokens,
                demo_tokens,
                target_tokens,
                tokenizer,
                device,
                salience_token_items=salience_token_items_tensor,
                use_original_loss=params.use_original_loss,
                use_salience_loss=params.use_salience_loss,
                use_weighted_salience_loss=params.use_weighted_salience_loss,
                trigger_position=params.trigger_position,
                model_type=params.model_type,
            )
            loss.backward()
            # stores gradients in extracted_gradients bc. of the hook we attached earlier
            # hence, loss can be deleted
            del loss, salience_token_items_tensor

        # Get average gradient w.r.t. the triggers.
        add_indices = [
            i for i, loss_type in enumerate(loss_types) if loss_type == "add"
        ]
        add_extracted_grads = []
        for i in add_indices:
            extracted_grad = extracted_grads[i]
            if (
                params.use_weighted_neg and demo_types[i] == "neg"
            ):  # Amplify neg associations.
                extracted_grad *= 2
            add_extracted_grads.append(extracted_grad)
        add_grad_tensor = torch.stack(add_extracted_grads)  # Convert to tensor.
        add_grad_tensor = torch.sum(add_grad_tensor, dim=0)  # Add all batches.
        add_grad_tensor = torch.sum(
            add_grad_tensor, dim=0
        )  # Add all samples in a `batch`.
        add_grad_tensor = add_grad_tensor[token_to_flip].unsqueeze(
            0
        )  # Use gradients at token_to_flip.
        grad = add_grad_tensor
        if params.use_dissociation_loss:
            grad = dissociate_grad(demo_types, grad, loss_types, params, token_to_flip)

        # Use hotflip (linear approximation) attack to get the top num_candidates.
        candidate_values, candidates = attacks.hotflip_attack(
            grad,
            embedding_weight,
            [trigger_tokens[right_counter_token_to_flip]],
            increase_loss=False,
            num_candidates=100,
        )
        candidates = candidates[0]
        candidate_values = candidate_values[0]

        find_best_candidate(
            params,
            device,
            model,
            tokenizer,
            trigger_tokens,
            all_items,
            new_beam_cache,
            candidates,
            candidate_values,
            neg_names,
            pos_names,
            neg_salience_token_items,
            pos_salience_token_items,
            right_counter_token_to_flip,
        )
    beam_cache = new_beam_cache
    return beam_cache, trigger_tokens


def find_best_candidate(
    params,
    device,
    model,
    tokenizer,
    trigger_tokens,
    all_items,
    new_beam_cache,
    candidates,
    candidate_values,
    neg_names,
    pos_names,
    neg_salience_token_items,
    pos_salience_token_items,
    right_counter_token_to_flip,
):
    # Try all the candidates and pick the best.
    loss_heap = []
    heapq.heapify(
        loss_heap
    )  # This is a min heap, so need to flip all losses to end up with the real smallest loss.
    eval_threshold = 5
    for cand_value, cand in zip(candidate_values, candidates):

        # Don't include tokens that have punctuation.
        decoded_cand = tokenizer.decode([cand])
        keep_token = keep_candidate_token(decoded_cand)
        if not keep_token:
            continue

        # replace one token with new candidate
        candidate_trigger_tokens = deepcopy(trigger_tokens)
        candidate_trigger_tokens[right_counter_token_to_flip] = cand
        curr_assoc_loss = 0.0
        curr_dissoc_loss = 0.0
        eval_set = collections.Counter()
        total_assoc_elements = 0.0
        total_dissoc_elements = 0.0
        for idx, (typ, demo_type, target_tokens) in enumerate(all_items):
            if eval_set[(typ, demo_type)] < eval_threshold:
                eval_set[(typ, demo_type)] += 1
            else:
                continue

            if demo_type == "neg":
                if params.neg_name_file:
                    demo_tokens = [tokenizer.encode("The " + n)[1:] for n in neg_names]
                else:
                    demo_tokens = tokenizer.encode(params.neg_demographic)
            elif demo_type == "pos":
                if params.pos_name_file:
                    demo_tokens = [tokenizer.encode("The " + n)[1:] for n in pos_names]
                else:
                    demo_tokens = tokenizer.encode(params.pos_demographic)
            else:
                raise NotImplementedError("Label is either neg or pos.")

            if params.use_salience_loss and not params.debias:
                if (demo_type == "neg" and typ == "add") or (
                    demo_type == "pos" and typ == "sub"
                ):
                    # association loss between neg demographic and neg samples (to add) or
                    # association loss between pos demographic and neg samples (to subtract).
                    salience_token_items = neg_salience_token_items
                elif (demo_type == "pos" and typ == "add") or (
                    demo_type == "neg" and typ == "sub"
                ):
                    # association loss between pos demographic and pos samples (to add) or
                    # association loss between neg demographic and pos samples (to subtract).
                    salience_token_items = pos_salience_token_items
                else:
                    raise NotImplementedError(
                        "Label and demographic pair not possible",
                        typ,
                        demo_type,
                    )
                # Add demo to salience token items.
                salience_token_items_tensor = torch.tensor(
                    salience_token_items,
                    device=device,
                    dtype=torch.long,
                )
            else:
                salience_token_items_tensor = None

            # get loss, update current best if its lower loss
            loss, mask_and_target = get_loss(
                model,
                params.batch_size,
                candidate_trigger_tokens,
                demo_tokens,
                target_tokens,
                tokenizer,
                device,
                salience_token_items=salience_token_items_tensor,
                use_original_loss=params.use_original_loss,
                use_salience_loss=params.use_salience_loss,
                use_weighted_salience_loss=params.use_weighted_salience_loss,
                trigger_position=params.trigger_position,
                model_type=params.model_type,
            )
            if typ == "add":
                # Losses are averaged per non-ignored element per sample per batch.
                # Since we are calculating overall loss over many batches, re-calc average.
                curr_num_elements = 0
                for sample in mask_and_target:
                    curr_num_elements += sum([1 for elem in sample if elem != -1])
                total_assoc_elements += curr_num_elements
                if (
                    demo_type == "neg" and params.use_weighted_neg
                ):  # Amplify neg associations.
                    curr_assoc_loss += 2 * loss.data.item() * curr_num_elements
                else:
                    curr_assoc_loss += loss.data.item() * curr_num_elements
            elif typ == "sub":
                curr_num_elements = 0
                for sample in mask_and_target:
                    curr_num_elements += sum([1 for elem in sample if elem != -1])
                total_dissoc_elements += curr_num_elements
                if (
                    demo_type == "neg" and params.use_weighted_neg
                ):  # Amplify neg associations.
                    curr_dissoc_loss += 2 * loss.data.item() * curr_num_elements
                else:
                    curr_dissoc_loss += loss.data.item() * curr_num_elements
            del loss, salience_token_items_tensor

            if all([x == eval_threshold for x in eval_set.values()]):
                break

        curr_assoc_loss /= total_assoc_elements
        if params.use_dissociation_loss:
            curr_dissoc_loss /= total_dissoc_elements
            curr_total_loss = (params.alpha * curr_assoc_loss) - (
                params.beta * curr_dissoc_loss
            )
        else:
            curr_total_loss = curr_assoc_loss

        # Keep top beam_size elements.
        # Note that beam_cache, new_beam_cache, and loss_heap all have reverse sign losses.
        curr_total_loss *= -1
        if len(new_beam_cache) < params.beam_size:
            heapq.heappush(loss_heap, curr_total_loss)
            new_beam_cache.append((curr_total_loss, deepcopy(candidate_trigger_tokens)))
            curr_worst_loss = heapq.nsmallest(1, loss_heap)[0]
        else:
            if curr_total_loss > curr_worst_loss:  # Remember, signs are flipped.
                # Kick out 1 trigger_tokens sequence with loss = curr_worst_loss.
                curr_worst_loss_idx_list = [
                    cache_idx
                    for cache_idx, (x, _) in enumerate(new_beam_cache)
                    if x == curr_worst_loss
                ]
                del new_beam_cache[curr_worst_loss_idx_list[0]]
                heapq.heappop(loss_heap)

                heapq.heappush(loss_heap, curr_total_loss)
                new_beam_cache.append(
                    (
                        curr_total_loss,
                        deepcopy(candidate_trigger_tokens),
                    )
                )
                curr_worst_loss = heapq.nsmallest(1, loss_heap)[0]


def dissociate_grad(demo_types, grad, loss_types, params, token_to_flip):
    global extracted_grads
    grad *= params.alpha
    sub_indices = [i for i, loss_type in enumerate(loss_types) if loss_type == "sub"]
    sub_extracted_grads = []
    for i in sub_indices:
        extracted_grad = extracted_grads[i]
        if (
            params.use_weighted_neg and demo_types[i] == "neg"
        ):  # Amplify neg associations.
            extracted_grad *= 2
        sub_extracted_grads.append(extracted_grad)
    sub_grad_tensor = torch.stack(sub_extracted_grads)  # Convert to tensor.
    sub_grad_tensor = torch.sum(sub_grad_tensor, dim=0)  # Add all batches.
    sub_grad_tensor = torch.sum(sub_grad_tensor, dim=0)  # Add all samples in a `batch`.
    sub_grad_tensor = sub_grad_tensor[token_to_flip].unsqueeze(
        0
    )  # Use gradients at token_to_flip.
    grad -= params.beta * sub_grad_tensor
    return grad


def keep_candidate_token(candidate):
    """Filter out undesired candidate tokens."""
    # Filter out candidates with punctuation and numbers.
    remove_punc = str.maketrans("", "", string.punctuation)
    new_candidate = candidate.translate(remove_punc)
    remove_digits = str.maketrans("", "", string.digits)
    new_candidate = new_candidate.translate(remove_digits)
    # Filter out byte tokens.
    if new_candidate.isprintable():
        return candidate == new_candidate
    else:
        return False
