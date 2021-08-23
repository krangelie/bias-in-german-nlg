import numpy as np
import torch

import src.constants as constants

# Gets the loss of the target_tokens using the triggers as the context.


def get_loss(
    language_model,
    batch_size,
    trigger_tokens,
    demo_tokens,
    target_tensor,
    tokenizer,
    device=torch.device("cuda"),
    salience_token_items=None,
    use_original_loss=True,
    use_salience_loss=False,
    use_weighted_salience_loss=False,
    trigger_position=constants.HEAD,
    model_type=constants.GPT2,
):
    trigger_tensor = (
        torch.tensor(trigger_tokens, device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    device = "cuda"
    complete_trigger_lengths = []

    respect_contexts = constants.GPT2_RCONTEXTS
    occupation_contexts = constants.GPT2_OCONTEXTS

    for sample_idx, target_token_sample in enumerate(target_tensor):
        num_prefix_tokens = len(trigger_tokens)
        # trigger_position = {head, body_demographic, body_biascontext}.
        if (
            constants.BODY in trigger_position
        ):  # Format: [demographic|biascontext] + trigger.
            if (
                constants.DEMO not in trigger_position
            ):  # Trigger does not include demographic yet.
                if type(demo_tokens[0]) == list:  # List of name token ID lists.
                    num_prefix_tokens += len(demo_tokens[sample_idx % len(demo_tokens)])
                else:  # Single list of demographic token IDs.
                    num_prefix_tokens += len(demo_tokens)
            complete_trigger_lengths.append(num_prefix_tokens)
        elif (
            constants.HEAD in trigger_position
        ):  # Format: trigger + demographic + bias_context.
            target_token_sample = [
                x for x in target_token_sample.tolist() if x != constants.PAD_TOKEN_ID
            ]
            target_str = tokenizer.decode(
                target_token_sample
            )  # Convert to string to find bias context strings.
            bias_context_tokens = None
            for c in respect_contexts + occupation_contexts:
                if model_type == constants.GPT2:
                    context_after = c.strip()
                    if context_after in target_str:
                        bias_context_tokens = tokenizer.encode("Das " + context_after)[
                            1:
                        ]  # Dummy first token so that the correct BPE token ID is used for the second token.
                        break

            if type(demo_tokens[0]) == list:  # List of name token ID lists.
                num_prefix_tokens += len(demo_tokens[sample_idx % len(demo_tokens)])
            else:
                num_prefix_tokens += len(demo_tokens)
            num_prefix_tokens += len(bias_context_tokens)
            complete_trigger_lengths.append(num_prefix_tokens)

    if constants.BODY in trigger_position:
        lm_input_list = [trigger_tensor]
        # For BODY trigger_positions, the trigger_tensor includes bias contexts.
        if (
            constants.DEMO not in trigger_position
        ):  # Trigger does not include demographic, we have to separately concat.
            if type(demo_tokens[0]) == list:  # List of name token ID lists.
                if len(demo_tokens) < batch_size:
                    extended_demo_tokens = []
                    idx = 0
                    while len(extended_demo_tokens) < batch_size:
                        extended_demo_tokens.append(demo_tokens[idx % len(demo_tokens)])
                        idx += 1
                else:
                    demo_tensor = torch.tensor(
                        demo_tokens[:batch_size], device=device, dtype=torch.long
                    )
            else:
                demo_tensor = (
                    torch.tensor(demo_tokens, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
            lm_input_list = [demo_tensor] + lm_input_list
        # target_tensor does not include demo or bias_contexts.
        lm_input_list += [target_tensor]
        lm_input = torch.cat(lm_input_list, dim=1)
    else:
        # target_tensor already includes demo + bias_contexts.
        lm_input = torch.cat(
            (trigger_tensor, target_tensor), dim=1
        )  # we feed the model the trigger + target texts

    mask_out = torch.ones_like(
        lm_input
    )  # prepare to mask loss for trigger + demo + bias context tokens.
    for sample_idx, sample in enumerate(mask_out):
        for tok_idx in range(complete_trigger_lengths[sample_idx]):
            sample[
                tok_idx
            ] = constants.PAD_TOKEN_ID  # Mask these trigger+other prefix tokens out.
    # mask_out = Use lm_input's end padding, mask_out's prefix padding and otherwise mask_out's 1's for target content.
    mask_out = torch.where(lm_input == constants.PAD_TOKEN_ID, lm_input, mask_out)
    mask_and_target = torch.where(
        mask_out == 1, lm_input, mask_out
    )  # -1...lm_input -1...
    lm_input[
        lm_input == constants.PAD_TOKEN_ID
    ] = 1  # put random token of 1 at end of context (it's masked out) # Format: target 1...

    # Printing for debugging.
    # print('trigger_tensor[0]', tokenizer.decode(trigger_tokens), trigger_tensor[0])
    # print('target_tensor[0]', target_tensor[0])
    # print('lm_input[0]', lm_input[0])
    # print('mask_and_target[0]', mask_and_target[0])

    if use_original_loss:
        loss = language_model(lm_input, labels=mask_and_target)[0]
    else:
        loss = None

    if use_salience_loss:
        # Create mask to mask out non-salient tokens.
        non_salience_mask_out = constants.PAD_TOKEN_ID * torch.ones_like(
            mask_and_target
        )

        if use_weighted_salience_loss:
            for x in range(5, 26):
                if (salience_token_items[mask_and_target] == x).byte().any():
                    non_salience_mask_and_target = torch.where(
                        salience_token_items[mask_and_target] == x,
                        mask_and_target,
                        non_salience_mask_out,
                    )
                    # Calculate salience loss.
                    salience_loss = language_model(
                        lm_input, labels=non_salience_mask_and_target
                    )[0]
                    del non_salience_mask_and_target

                    # Combine normal loss and salience loss.
                    if loss is None:
                        loss = salience_loss * float(x)
                    else:
                        loss += salience_loss * float(x)
                    del salience_loss

        else:  # Calculate unweighted salience loss.
            if (salience_token_items[mask_and_target] > 0).byte().any():
                non_salience_mask_and_target = torch.where(
                    salience_token_items[mask_and_target] > 0,
                    mask_and_target,
                    non_salience_mask_out,
                )
                # Calculate salience loss.
                salience_loss = language_model(
                    lm_input, labels=non_salience_mask_and_target
                )[0]
                del non_salience_mask_and_target

                # Combine normal loss and salience loss.
                if loss is None:
                    loss = salience_loss
                else:
                    loss += salience_loss
                del salience_loss

    return loss, mask_and_target


# Creates the batch of target texts with pad token placed at the end of the sequences for padding (for masking out the loss).
def make_target_batch(tokenizer, device, target_texts, max_len, batch_size):
    # encode items and get the max length
    encoded_texts = []
    for idx, target_text in enumerate(target_texts):
        encoded_target_text = tokenizer.encode(target_text)
        encoded_texts.append(encoded_target_text)

    # pad tokens, i.e., append pad_token_id to the end of the non-longest ones
    for indx, encoded_text in enumerate(encoded_texts):
        if len(encoded_text) < max_len:
            encoded_texts[indx].extend(
                [constants.PAD_TOKEN_ID] * (max_len - len(encoded_text))
            )
        elif len(encoded_text) > max_len:
            encoded_texts[indx] = encoded_text[:max_len]

    # convert to tensors and batch them up
    target_tokens_batch = None
    for encoded_text in encoded_texts:
        target_tokens = torch.tensor(
            encoded_text, device=device, dtype=torch.long
        ).unsqueeze(0)
        if target_tokens_batch is None:
            target_tokens_batch = target_tokens
            if target_tokens_batch.shape[0] == batch_size:
                yield target_tokens_batch
                target_tokens_batch = None
        elif target_tokens_batch.shape[0] < batch_size:
            target_tokens_batch = torch.cat((target_tokens_batch, target_tokens), dim=0)
            if target_tokens_batch.shape[0] == batch_size:
                yield target_tokens_batch
                target_tokens_batch = None

    # Just drop the extra samples.
    # if target_tokens_batch is not None:
    #     yield target_tokens_batch


def get_target_losses_and_threshold(
    params,
    tokenizer,
    neg_demo_neg_target_texts,
    neg_demo_neu_target_texts,
    neg_demo_pos_target_texts,
    pos_demo_neg_target_texts,
    pos_demo_neu_target_texts,
    pos_demo_pos_target_texts,
):
    device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 50

    # batch and pad the target tokens
    neg_demo_neg_target_tokens_gen = make_target_batch(
        tokenizer, device, neg_demo_neg_target_texts, max_len, params.batch_size
    )
    pos_demo_pos_target_tokens_gen = make_target_batch(
        tokenizer, device, pos_demo_pos_target_texts, max_len, params.batch_size
    )
    neg_demo_neg_target_tokens_gen = list(neg_demo_neg_target_tokens_gen)
    same_demo_target_threshold = len(neg_demo_neg_target_tokens_gen)
    pos_demo_pos_target_tokens_gen = list(pos_demo_pos_target_tokens_gen)
    same_demo_target_losses = (
        neg_demo_neg_target_tokens_gen + pos_demo_pos_target_tokens_gen
    )
    if params.use_dissociation_loss:
        pos_demo_neg_target_tokens_gen = make_target_batch(
            tokenizer, device, pos_demo_neg_target_texts, max_len, params.batch_size
        )
        neg_demo_pos_target_tokens_gen = make_target_batch(
            tokenizer, device, neg_demo_pos_target_texts, max_len, params.batch_size
        )
        pos_demo_neg_target_tokens_gen = list(pos_demo_neg_target_tokens_gen)
        diff_demo_target_threshold = len(pos_demo_neg_target_tokens_gen)
        neg_demo_pos_target_tokens_gen = list(neg_demo_pos_target_tokens_gen)
        diff_demo_target_losses = (
            pos_demo_neg_target_tokens_gen + neg_demo_pos_target_tokens_gen
        )
    neu_target_losses = []
    if params.neu_sample_file:
        pos_demo_neu_target_tokens_gen = make_target_batch(
            tokenizer, device, pos_demo_neu_target_texts, max_len, params.batch_size
        )
        neg_demo_neu_target_tokens_gen = make_target_batch(
            tokenizer, device, neg_demo_neu_target_texts, max_len, params.batch_size
        )
        pos_demo_neu_target_tokens_gen = list(pos_demo_neu_target_tokens_gen)
        neu_target_threshold = len(pos_demo_neu_target_tokens_gen)
        neg_demo_neu_target_tokens_gen = list(neg_demo_neu_target_tokens_gen)
        neu_target_losses = (
            pos_demo_neu_target_tokens_gen + neg_demo_neu_target_tokens_gen
        )
    return (
        diff_demo_target_losses,
        diff_demo_target_threshold,
        neu_target_losses,
        neu_target_threshold,
        same_demo_target_losses,
        same_demo_target_threshold,
    )


def interleave_losses(
    diff_demo_target_losses,
    diff_demo_target_threshold,
    neu_target_losses,
    neu_target_threshold,
    params,
    same_demo_target_losses,
    same_demo_target_threshold,
):
    # Interleave negative and positive add_losses, shuffle all items.
    all_items = []
    if params.debias:  # Generate debiasing triggers.
        assert neu_target_losses
        for idx, l in enumerate(neu_target_losses):
            if idx < neu_target_threshold:
                all_items += [("add", "pos", l)]
            else:
                all_items += [("add", "neg", l)]
        if params.debias == 1:
            # A - B where A = neu_target_losses and B = same_demo_target_losses + diff_demo_target_losses.
            same_demo_target_loss_type = "sub"
            diff_demo_target_loss_type = "sub"
    else:  # Debias = 0, generate adversarial triggers.
        same_demo_target_loss_type = "add"
        diff_demo_target_loss_type = "sub"
    for idx, l in enumerate(same_demo_target_losses):
        if params.num_demographics == 1:
            if idx < same_demo_target_threshold:
                # (Whether to add or subtract loss (add), demographic type (neg), samples).
                all_items += [(same_demo_target_loss_type, "neg", l)]
        elif params.num_demographics == 2:
            if idx < same_demo_target_threshold:
                if params.debias == 2:
                    # A - B where A = neu_target_losses + pos_target_losses, and B = neg_target_losses.
                    same_demo_target_loss_type = "sub"
                all_items += [
                    (same_demo_target_loss_type, "neg", l)
                ]  # (Whether to add or subtract loss, demographic type, samples).
            else:
                if params.debias == 2:
                    same_demo_target_loss_type = "add"
                all_items += [(same_demo_target_loss_type, "pos", l)]
        else:
            raise NotImplementedError(
                "num_demographics has to be in [1, 2]: %s" % params.num_demographics
            )
    if params.use_dissociation_loss:
        for idx, l in enumerate(diff_demo_target_losses):
            if idx < diff_demo_target_threshold:
                if params.debias == 2:
                    diff_demo_target_loss_type = "sub"
                all_items += [(diff_demo_target_loss_type, "pos", l)]
            else:
                if params.debias == 2:
                    diff_demo_target_loss_type = "add"
                all_items += [(diff_demo_target_loss_type, "neg", l)]
    np.random.shuffle(all_items)
    # Useful for debugging:
    # for i in range(min(10, len(all_items))):
    #     itm = all_items[i]
    #     sample = [x for x in itm[2][0].tolist() if x != constants.PAD_TOKEN_ID]
    #     print(sample)
    #     print(itm[0], itm[1], tokenizer.decode(sample))
    return all_items
