import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=512,
    block_length=512,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="core",
    mask_id=None,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    remask_k=1,
    revise_every=8,
    candidate_m=32,
    base_masking="confidence",
    joint_reeval=False,
    eos_token_id=None,
    eot_token_id=None,
):
    if mask_id is None:
        raise ValueError("mask_id must be provided for CORE-style generation")

    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by number of blocks ({num_blocks})"
        )

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            remask_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            remask_counts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf and eos_token_id is not None:
                logits[:, :, eos_token_id] = -torch.inf
            if confidence_eos_eot_inf:
                if eos_token_id is not None:
                    logits[:, :, eos_token_id] = -torch.inf
                if eot_token_id is not None:
                    logits[:, :, eot_token_id] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_prob = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                -1,
            )

            if remasking in ["core", "margin_remask", "random_remask"]:
                if base_masking == "confidence":
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                        -1,
                    )
                elif base_masking == "margin":
                    top2_vals, _ = p.topk(2, dim=-1)
                    x0_p = top2_vals[..., 0] - top2_vals[..., 1]
                else:
                    raise ValueError(f"Unsupported base_masking: {base_masking}")

                window_start = int(0.25 * steps_per_block)
                window_end = int(0.75 * steps_per_block)
                verify_now = (
                    revise_every is not None
                    and revise_every > 0
                    and i % revise_every == 0
                    and window_start <= i < window_end
                )

                if verify_now:
                    top2_vals, _ = p.topk(2, dim=-1)
                    margin = top2_vals[..., 0] - top2_vals[..., 1]
                    pick_score = -margin

                    in_block = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    in_block[:, block_start:block_end] = True

                    k_protect = 16
                    protect = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    protect[:, max(block_start, block_end - k_protect) : block_end] = True

                    candidates = (~mask_index) & in_block & (~prompt_index) & (~protect)

                    verify_mask1 = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    for b in range(x.shape[0]):
                        cand_b = torch.nonzero(candidates[b], as_tuple=False).squeeze(-1)
                        if cand_b.numel() == 0:
                            continue
                        m = min(int(candidate_m), int(cand_b.numel()))
                        scores_b = pick_score[b, cand_b]
                        _, idx1 = torch.topk(scores_b, k=m)
                        verify_mask1[b, cand_b[idx1]] = True

                    def _run_verify_pass(x_in, verify_mask_local):
                        x_filled = x_in.clone()
                        x_ver = x_in.clone()
                        x_ver[verify_mask_local] = mask_id

                        if cfg_scale > 0.0:
                            un_x = x_ver.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x_ver, un_x], dim=0)
                            if attention_mask is not None:
                                attention_mask_ = torch.cat(
                                    [attention_mask, attention_mask], dim=0
                                )
                            else:
                                attention_mask_ = None
                            logits_ = model(x_, attention_mask=attention_mask_).logits
                            logits_c, un_logits = torch.chunk(logits_, 2, dim=0)
                            logits_ver = un_logits + (cfg_scale + 1) * (
                                logits_c - un_logits
                            )
                        else:
                            logits_ver = model(x_ver, attention_mask=attention_mask).logits

                        if logits_eos_inf and eos_token_id is not None:
                            logits_ver[:, :, eos_token_id] = -torch.inf
                        if confidence_eos_eot_inf:
                            if eos_token_id is not None:
                                logits_ver[:, :, eos_token_id] = -torch.inf
                            if eot_token_id is not None:
                                logits_ver[:, :, eot_token_id] = -torch.inf

                        p_ver = F.softmax(logits_ver, dim=-1)
                        tok_prob = torch.squeeze(
                            torch.gather(
                                p_ver, dim=-1, index=torch.unsqueeze(x_filled, -1)
                            ),
                            -1,
                        )
                        pll = torch.log(tok_prob + 1e-10)

                        logits_with_noise_ver = add_gumbel_noise(
                            logits_ver, temperature=0.0
                        )
                        x0_ver = torch.argmax(logits_with_noise_ver, dim=-1)
                        x0_prob_ver = torch.squeeze(
                            torch.gather(
                                p_ver, dim=-1, index=torch.unsqueeze(x0_ver, -1)
                            ),
                            -1,
                        )

                        if base_masking == "confidence":
                            x0_p_ver = torch.squeeze(
                                torch.gather(
                                    p_ver, dim=-1, index=torch.unsqueeze(x0_ver, -1)
                                ),
                                -1,
                            )
                        elif base_masking == "margin":
                            top2v, _ = p_ver.topk(2, dim=-1)
                            x0_p_ver = top2v[..., 0] - top2v[..., 1]
                        else:
                            raise ValueError(
                                f"Unsupported base_masking: {base_masking}"
                            )

                        return pll, x0_ver, x0_p_ver, x0_prob_ver, p_ver, x_filled

                    def _score_from_verify(
                        score_mode, pll, p_ver, x0_ver, x0_prob_ver, x_filled
                    ):
                        if score_mode == "core":
                            wants_change = x0_ver != x_filled
                            repl_conf_ok = x0_prob_ver >= 0.3
                            return torch.where(
                                wants_change & repl_conf_ok,
                                -pll,
                                torch.full_like(pll, -float("inf")),
                            )
                        if score_mode == "margin_remask":
                            top2v, _ = p_ver.topk(2, dim=-1)
                            margin_ver = top2v[..., 0] - top2v[..., 1]
                            return -margin_ver
                        if score_mode == "random_remask":
                            return torch.rand_like(pll)
                        raise ValueError(f"Unknown score_mode: {score_mode}")

                    if verify_mask1.any().item():
                        pll1, x0_ver1, x0_p_ver1, x0_prob_ver1, p_ver1, x_filled1 = (
                            _run_verify_pass(x, verify_mask1)
                        )

                        remask_score1 = _score_from_verify(
                            remasking,
                            pll1,
                            p_ver1,
                            x0_ver1,
                            x0_prob_ver1,
                            x_filled1,
                        )
                        remask_score1 = torch.where(
                            verify_mask1,
                            remask_score1,
                            torch.full_like(remask_score1, -float("inf")),
                        )

                        for b in range(x.shape[0]):
                            f_step = int(num_transfer_tokens[b, i].item())
                            if f_step <= 0:
                                continue

                            max_remask = min(remask_k, f_step)
                            if max_remask <= 0:
                                continue

                            scores_b = remask_score1[b]
                            finite = torch.isfinite(scores_b)
                            if not finite.any():
                                continue

                            k_remask = min(max_remask, int(finite.sum().item()))
                            if k_remask <= 0:
                                continue

                            masked_scores = scores_b.masked_fill(~finite, float("-inf"))
                            _, idx = torch.topk(masked_scores, k=k_remask)
                            remask_index[b, idx] = True
                            remask_counts[b] = k_remask

                        x0 = torch.where(remask_index, x0_ver1, x0)
                        x0_p = torch.where(remask_index, x0_p_ver1, x0_p)
                        x0_prob = torch.where(remask_index, x0_prob_ver1, x0_prob)

                        x[remask_index] = mask_id
                        mask_index = x == mask_id

                        if joint_reeval and remask_index.any().item():
                            force_ok_mask = remask_index & torch.isfinite(x0_prob_ver1)
                            x_joint = x.clone()
                            x_joint[force_ok_mask] = x0_ver1[force_ok_mask]

                            if cfg_scale > 0.0:
                                un_x_j = x_joint.clone()
                                un_x_j[prompt_index] = mask_id
                                x_j_cat = torch.cat([x_joint, un_x_j], dim=0)
                                if attention_mask is not None:
                                    att_j_cat = torch.cat(
                                        [attention_mask, attention_mask], dim=0
                                    )
                                else:
                                    att_j_cat = None
                                logits_j_cat = model(
                                    x_j_cat, attention_mask=att_j_cat
                                ).logits
                                logits_j_c, un_logits_j = torch.chunk(
                                    logits_j_cat, 2, dim=0
                                )
                                logits_joint = un_logits_j + (cfg_scale + 1) * (
                                    logits_j_c - un_logits_j
                                )
                            else:
                                logits_joint = model(
                                    x_joint, attention_mask=attention_mask
                                ).logits

                            if logits_eos_inf and eos_token_id is not None:
                                logits_joint[:, :, eos_token_id] = -torch.inf
                            if confidence_eos_eot_inf:
                                if eos_token_id is not None:
                                    logits_joint[:, :, eos_token_id] = -torch.inf
                                if eot_token_id is not None:
                                    logits_joint[:, :, eot_token_id] = -torch.inf

                            p_joint = F.softmax(logits_joint, dim=-1)
                            logits_with_noise_joint = add_gumbel_noise(
                                logits_joint, temperature=temperature
                            )
                            x0_joint = torch.argmax(logits_with_noise_joint, dim=-1)

                            if base_masking == "confidence":
                                x0_p_joint = torch.squeeze(
                                    torch.gather(
                                        p_joint,
                                        dim=-1,
                                        index=torch.unsqueeze(x0_joint, -1),
                                    ),
                                    -1,
                                )
                            elif base_masking == "margin":
                                top2v_j, _ = p_joint.topk(2, dim=-1)
                                x0_p_joint = top2v_j[..., 0] - top2v_j[..., 1]
                            else:
                                raise ValueError(
                                    f"Unsupported base_masking: {base_masking}"
                                )

                            is_remaining_masked = mask_index & (~remask_index)
                            x0 = torch.where(is_remaining_masked, x0_joint, x0)
                            x0_p = torch.where(is_remaining_masked, x0_p_joint, x0_p)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -torch.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            for j in range(confidence.shape[0]):
                f_step = int(num_transfer_tokens[j, i].item())
                if f_step <= 0:
                    continue

                rmask_j = remask_index[j]
                r_count = int(remask_counts[j].item())
                k = f_step + r_count
                if k <= 0:
                    continue

                force_ok = rmask_j & torch.isfinite(x0_prob[j])
                transfer_index[j, force_ok] = True
                forced = int(force_ok.sum().item())

                remaining = k - forced
                if remaining <= 0:
                    continue

                scores_j = confidence[j].clone()
                scores_j[force_ok] = -float("inf")

                finite = torch.isfinite(scores_j)
                if not finite.any():
                    continue

                remaining = min(remaining, int(finite.sum().item()))
                _, select_index = torch.topk(scores_j, k=remaining)
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    gen_start = prompt.shape[1]
    if not (x[:, gen_start:] != mask_id).all():
        raise AssertionError("Some positions are still masked at the end of generation")

    return x
