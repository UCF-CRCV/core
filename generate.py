import os
import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=512,
    block_length=512,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    remask_k=1
):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length),
                                                               dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    stats = {
        "core_calls": 0,
        "verified_tokens": 0,
        "remasked_tokens": 0,
        "changed_tokens": 0,
        "core_sum_verified": 0.0,
        "core_sum_remasked": 0.0
    }

    mech = {
        "core_verified_kept": [],
        "core_verified_remasked": [],
        "remasked_changed": [],
        "remasked_unchanged": [],
        "delta_remasked": [],
        "margin_verified_kept": [],
        "margin_verified_remasked": []
    }

    MECH_CAP = int(os.environ.get("MECH_CAP", "4096"))  # max values appended per verify call per bucket

    def _push(bucket, vals_1d):
        # vals_1d: 1D tensor on any device
        if vals_1d is None or vals_1d.numel() == 0:
            return
        vals_1d = vals_1d.detach().flatten()
        if vals_1d.numel() > MECH_CAP:
            idx = torch.randperm(vals_1d.numel(), device=vals_1d.device)[:MECH_CAP]
            vals_1d = vals_1d[idx]
        mech[bucket].append(vals_1d.float().cpu())

    def _push_pair(bucket_a, bucket_b, a_1d, b_1d):
        # a_1d and b_1d are 1D tensors with matched entries (same tokens)
        if a_1d is None or b_1d is None or a_1d.numel() == 0:
            return
        a = a_1d.detach().flatten()
        b = b_1d.detach().flatten()
        if a.numel() == 0:
            return
        assert a.numel() == b.numel(), "Paired logging requires same number of elements"

        if a.numel() > MECH_CAP:
            idx = torch.randperm(a.numel(), device=a.device)[:MECH_CAP]
            a = a[idx]
            b = b[idx]

        mech[bucket_a].append(a.float().cpu())
        mech[bucket_b].append(b.float().cpu())


    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)
            
            remask_index  = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            remask_counts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf
            if confidence_eos_eot_inf:
                logits[:, :, 126081] = -torch.inf
                logits[:, :, 126348] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0_prob = x0_p
            
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                x0_prob = torch.full_like(x0_p, float('nan'))

            elif remasking == 'topk_margin':
                # top-k margin certainty
                # cert(i) = p1(i) - p2(i), where p1 >= p2 are the top-2 probs at that position
                p = F.softmax(logits, dim=-1)                     # [B, L, V]
                top2_vals, _ = p.topk(2, dim=-1)                  # [B, L, 2]
                x0_p = top2_vals[..., 0] - top2_vals[..., 1]      # [B, L]
                x0_prob = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            elif remasking in ["core", "margin_remask", "random_remask"]:
                score_mode = remasking
                base_masking = os.environ.get("BASE_MASKING", "confidence")
                assert base_masking in ["confidence", "margin"]
            
                # --- knobs ---
                revise_every = int(os.environ.get("REVISE_EVERY", "8")) # run revision every N inner steps (0 disables)
                candidate_m  = int(os.environ.get("CANDIDATE_M", "32")) # how many filled tokens to verify per sample when verifying

                B, L = x.shape

                # Base distribution at current real step (used for normal fill confidence)
                p = F.softmax(logits, dim=-1)  # [B, L, V]
                x0_prob = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                if base_masking == "confidence":
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                        -1
                    )  # [B, L]
                elif base_masking == "margin":
                    top2_vals, _ = p.topk(2, dim=-1)
                    x0_p = top2_vals[..., 0] - top2_vals[..., 1]   # [B,L]

                window_start = int(0.25 * steps)
                window_end   = int(0.75 * steps)                    
                verify_now = (
                    (revise_every is not None) and (revise_every > 0)
                    and (i % revise_every == 0)
                    and (window_start <= i < window_end)
                )

                if verify_now:
                    stats["core_calls"] += 1

                    # Pick candidate tokens to verify (cheap) using margin under current logits
                    top2_vals, _ = p.topk(2, dim=-1)                # [B, L, 2]
                    margin = top2_vals[..., 0] - top2_vals[..., 1]  # [B, L]
                    pick_score = -margin                            # smaller margin => larger score
                    
                    # -------------------------------------------------

                    # Candidate positions: already-filled, in-block, non-prompt, non-protected
                    in_block = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    in_block[:, block_start:block_end] = True

                    k_protect = 16
                    protect = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    protect[:, max(block_start, block_end - k_protect):block_end] = True

                    candidates = (~mask_index) & in_block & (~prompt_index) & (~protect)

                    # Build verify mask: choose up to candidate_m positions per sample
                    verify_mask1 = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    for b in range(B):
                        cand_b = torch.nonzero(candidates[b], as_tuple=False).squeeze(-1)
                        if cand_b.numel() == 0:
                            continue
                        m = min(int(candidate_m), int(cand_b.numel()))
                        scores_b = pick_score[b, cand_b]
                        
                        _, idx1 = torch.topk(scores_b, k=m)
                        verify_mask1[b, cand_b[idx1]] = True
                    
                    verify_mask = verify_mask1
                    stats["verified_tokens"] += int(verify_mask.sum().item())

                    def _run_verify_pass(x_in, verify_mask_local):
                        x_filled = x_in.clone()
                        x_ver = x_in.clone()
                        x_ver[verify_mask_local] = mask_id

                        # CFG-aware forward
                        if cfg_scale > 0.:
                            un_x = x_ver.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x_ver, un_x], dim=0)
                            attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                            logits_ = model(x_, attention_mask=attention_mask_).logits
                            logits_c, un_logits = torch.chunk(logits_, 2, dim=0)
                            logits_ver = un_logits + (cfg_scale + 1) * (logits_c - un_logits)
                        else:
                            logits_ver = model(x_ver, attention_mask=attention_mask).logits

                        # EOS/EOT masking (mirror main)
                        if logits_eos_inf:
                            logits_ver[:, :, 126081] = -torch.inf
                        if confidence_eos_eot_inf:
                            logits_ver[:, :, 126081] = -torch.inf
                            logits_ver[:, :, 126348] = -torch.inf

                        p_ver = F.softmax(logits_ver, dim=-1)

                        tok_prob = torch.squeeze(
                            torch.gather(p_ver, dim=-1, index=torch.unsqueeze(x_filled, -1)),
                            -1
                        )
                        pll = torch.log(tok_prob + 1e-10)

                        logits_with_noise_ver = add_gumbel_noise(logits_ver, temperature=0.0)
                        x0_ver = torch.argmax(logits_with_noise_ver, dim=-1)
                        x0_prob_ver = torch.squeeze(torch.gather(p_ver, dim=-1, index=torch.unsqueeze(x0_ver, -1)), -1)

                        if base_masking == "confidence":
                            x0_p_ver = torch.squeeze(torch.gather(p_ver, dim=-1, index=torch.unsqueeze(x0_ver, -1)), -1)
                        elif base_masking == "margin":
                            top2v, _ = p_ver.topk(2, dim=-1)
                            x0_p_ver = top2v[..., 0] - top2v[..., 1]
                        elif base_masking == "random":
                            x0_p_ver = torch.rand((B, L), device=x.device)

                        return pll, x0_ver, x0_p_ver, x0_prob_ver, p_ver, x_filled

                    def _score_from_verify(score_mode, pll, p_ver, x0_ver, x0_prob_ver, x_filled):
                        # returns remask_score over ALL positions (not yet restricted to verify_mask)
                        if score_mode == "core":
                            wants_change = (x0_ver != x_filled)
                            repl_conf_ok = (x0_prob_ver >= 0.3)
                            remask_score = torch.where(
                                wants_change & repl_conf_ok,
                                -pll,
                                torch.full_like(pll, -float("inf"))
                            )

                        elif score_mode == "margin_remask":
                            top2v, _ = p_ver.topk(2, dim=-1)
                            margin_ver = top2v[..., 0] - top2v[..., 1]
                            remask_score = -margin_ver

                        elif score_mode == "random_remask":
                            remask_score = torch.rand_like(pll)

                        else:
                            raise ValueError(f"Unknown score_mode: {score_mode}")

                        return remask_score
            
                    if verify_mask1.any().item():
                        # ---------------- Pass 1 (original behavior) ----------------
                        pll1, x0_ver1, x0_p_ver1, x0_prob_ver1, p_ver1, x_filled1 = _run_verify_pass(x, verify_mask1)

                        # token-weighted verified PLL stats (only meaningful for revise)
                        if score_mode == "core":
                            stats["core_sum_verified"] += float(pll1[verify_mask1].sum().item())

                        remask_score1 = _score_from_verify(score_mode, pll1, p_ver1, x0_ver1, x0_prob_ver1, x_filled1)
                        remask_score1 = torch.where(
                            verify_mask1,
                            remask_score1,
                            torch.full_like(remask_score1, -float("inf"))
                        )

                        remask_score = remask_score1
                        verify_mask = verify_mask1

                        stats["verified_tokens"] += int(verify_mask1.sum().item())

                        # ---------------- Choose up to remask_k positions to remask ----------------
                        for b in range(B):
                            F_step = int(num_transfer_tokens[b, i].item())
                            if F_step <= 0:
                                continue

                            max_remask = min(remask_k, F_step)
                            if max_remask <= 0:
                                continue

                            scores_b = remask_score[b]
                            finite = torch.isfinite(scores_b)
                            print(f"[budget] step={i} b={b} F_step={F_step} max_remask={max_remask} finite={int(finite.sum().item())}")

                            if not finite.any():
                                continue

                            k_remask = min(max_remask, int(finite.sum().item()))
                            if k_remask <= 0:
                                continue

                            masked_scores = scores_b.masked_fill(~finite, float("-inf"))
                            _, idx = torch.topk(masked_scores, k=k_remask)
                            remask_index[b, idx] = True
                            remask_counts[b] = k_remask

                        r = remask_index
                        stats["remasked_tokens"] += int(r.sum().item())

                        # Use pass1 objects for refill and logging
                        pll = pll1
                        p_ver = p_ver1
                        x0_ver = x0_ver1
                        x0_p_ver = x0_p_ver1
                        x0_prob_ver = x0_prob_ver1
                        x_filled = x_filled1

                        if score_mode == "core" and r.any().item():
                            stats["core_sum_remasked"] += float(pll[r].sum().item())
                            stats["changed_tokens"] += int((x0_ver[r] != x_filled[r]).sum().item())

                        if remask_index.any().item():
                            for b in range(B):
                                idxs = torch.nonzero(remask_index[b], as_tuple=False).squeeze(-1).tolist()
                                for pos in idxs:
                                    old = int(x_filled[b, pos].item())
                                    new = int(x0_ver[b, pos].item())
                                    changed = (old != new)
                                    print(f"[remask] step={i} b={b} pos={pos} old={old} new={new} changed={changed}")

                        # ---------------- mechanism logging ----------------
                        rem_mask  = verify_mask & remask_index
                        keep_mask = verify_mask & (~remask_index)

                        _push_pair("core_verified_kept", "margin_verified_kept",
                                pll[keep_mask], margin[keep_mask])

                        _push_pair("core_verified_remasked", "margin_verified_remasked",
                                pll[rem_mask], margin[rem_mask])

                        if rem_mask.any():
                            changed_mask = rem_mask & (x0_ver != x_filled)
                            unchanged_mask = rem_mask & (x0_ver == x_filled)
                            _push("remasked_changed", pll[changed_mask])
                            _push("remasked_unchanged", pll[unchanged_mask])

                            new_prob = torch.squeeze(
                                torch.gather(p_ver, dim=-1, index=torch.unsqueeze(x0_ver, -1)),
                                -1
                            )
                            pll_new = torch.log(new_prob + 1e-10)
                            delta = pll_new - pll
                            _push("delta_remasked", delta[rem_mask])

                        # --- refill remasked tokens using masked-context logits (pass1) ---
                        x0   = torch.where(remask_index, x0_ver, x0)
                        x0_p = torch.where(remask_index, x0_p_ver, x0_p)
                        x0_prob = torch.where(remask_index, x0_prob_ver, x0_prob)

                        x[remask_index] = mask_id
                        mask_index = (x == mask_id)

            else:
                raise NotImplementedError(remasking)

            # prevent selecting beyond current block end
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            for j in range(confidence.shape[0]):
                _F = int(num_transfer_tokens[j, i].item())
                if _F <= 0:
                    continue

                rmask_j = remask_index[j]
                R = int(remask_counts[j].item())
                K = _F + R
                if K <= 0:
                    continue

                # 1) Force-commit ONLY remasked positions whose replacement is confident.
                force_ok = rmask_j & torch.isfinite(x0_prob[j])
                transfer_index[j, force_ok] = True
                forced = int(force_ok.sum().item())

                # 2) Spend remaining budget by confidence, excluding already-forced positions.
                remaining = K - forced
                if remaining <= 0:
                    continue

                scores_j = confidence[j].clone()
                scores_j[force_ok] = -float("inf")  # don't reselect already committed tokens

                finite = torch.isfinite(scores_j)
                if not finite.any():
                    continue

                remaining = min(remaining, int(finite.sum().item()))
                _, select_index = torch.topk(scores_j, k=remaining)
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    gen_start = prompt.shape[1]
    assert (x[:, gen_start:] != mask_id).all(), "Some positions are still masked at the end of generation!"
    
    if stats["core_calls"] > 0:
        calls = stats["core_calls"]
        avg_pll_verified = stats["core_sum_verified"] / max(1, stats["verified_tokens"])
        avg_pll_remasked = stats["core_sum_remasked"] / max(1, stats["remasked_tokens"])
        print(
            f"[revise] calls={calls} "
            f"verified/call={stats['verified_tokens']/calls:.1f} "
            f"remasked/call={stats['remasked_tokens']/calls:.2f} "
            f"changed/remasked={(stats['changed_tokens']/max(1,stats['remasked_tokens'])):.2f} "
            f"avg_pll_verified={avg_pll_verified:.3f} "
            f"avg_pll_remasked={avg_pll_remasked:.3f}"
        )
    else:
        print('revise_calls is zero')
    
    return x, stats, mech
