import torch
import torch.nn.functional as F


class DreamCoreRemaskHook:
    def __init__(
        self,
        model,
        input_width,
        max_length,
        prompt_attention_mask,
        mask_token_id,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="core",
        remask_k=1,
        revise_every=8,
        candidate_m=32,
        base_masking="confidence",
        joint_reeval=False,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        eos_token_id=None,
        eot_token_id=None,
        total_steps=128,
        protect_tail=16,
    ):
        self.model = model
        self.input_width = input_width
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.remask_k = remask_k
        self.revise_every = revise_every
        self.candidate_m = candidate_m
        self.base_masking = base_masking
        self.joint_reeval = joint_reeval
        self.logits_eos_inf = logits_eos_inf
        self.confidence_eos_eot_inf = confidence_eos_eot_inf
        self.eos_token_id = eos_token_id
        self.eot_token_id = eot_token_id
        self.total_steps = total_steps
        self.protect_tail = protect_tail
        self.prompt_attention_mask = prompt_attention_mask
        self._prepared_attention_mask, self._tok_idx = self._prepare_attention(
            prompt_attention_mask
        )

    def _prepare_attention(self, attention_mask):
        if attention_mask is None or not torch.any(attention_mask == 0.0):
            return "full", None

        padded = F.pad(
            attention_mask,
            (0, self.max_length - attention_mask.shape[1]),
            value=1.0,
        )
        tok_idx = padded.long().cumsum(-1) - 1
        tok_idx.masked_fill_(padded == 0, 1)
        full_mask = torch.logical_and(
            padded.unsqueeze(1).unsqueeze(-2),
            padded.unsqueeze(1).unsqueeze(-1),
        )
        return full_mask, tok_idx

    def _apply_special_masks(self, logits):
        if self.logits_eos_inf and self.eos_token_id is not None:
            logits[:, :, self.eos_token_id] = -torch.inf
        if self.confidence_eos_eot_inf:
            if self.eos_token_id is not None:
                logits[:, :, self.eos_token_id] = -torch.inf
            if self.eot_token_id is not None:
                logits[:, :, self.eot_token_id] = -torch.inf
        return logits

    def _forward_shifted_logits(self, x):
        logits = self.model(x, self._prepared_attention_mask, self._tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return self._apply_special_masks(logits)

    def _run_verify_pass(self, x_in, verify_mask):
        x_filled = x_in.clone()
        x_ver = x_in.clone()
        x_ver[verify_mask] = self.mask_token_id

        logits_ver = self._forward_shifted_logits(x_ver)
        p_ver = F.softmax(logits_ver, dim=-1)
        tok_prob = torch.gather(p_ver, dim=-1, index=x_filled.unsqueeze(-1)).squeeze(-1)
        pll = torch.log(tok_prob + 1e-10)

        # The verification pass should be deterministic; the original CORE
        # reference uses a zero-temperature re-evaluation here.
        x0_ver = torch.argmax(logits_ver, dim=-1)

        x0_prob_ver = torch.gather(p_ver, dim=-1, index=x0_ver.unsqueeze(-1)).squeeze(-1)

        if self.base_masking == "confidence":
            x0_p_ver = x0_prob_ver
        elif self.base_masking == "margin":
            top2v, _ = p_ver.topk(2, dim=-1)
            x0_p_ver = top2v[..., 0] - top2v[..., 1]
        else:
            raise ValueError(f"Unsupported base_masking: {self.base_masking}")

        return pll, x0_ver, x0_p_ver, x0_prob_ver, p_ver, x_filled

    def _score_from_verify(self, pll, p_ver, x0_ver, x0_prob_ver, x_filled):
        if self.remasking == "core":
            wants_change = x0_ver != x_filled
            repl_conf_ok = x0_prob_ver >= 0.3
            return torch.where(
                wants_change & repl_conf_ok,
                -pll,
                torch.full_like(pll, -float("inf")),
            )
        if self.remasking == "margin_remask":
            top2v, _ = p_ver.topk(2, dim=-1)
            margin_ver = top2v[..., 0] - top2v[..., 1]
            return -margin_ver
        if self.remasking == "random_remask":
            return torch.rand_like(pll)
        raise ValueError(f"Unknown remasking mode: {self.remasking}")

    def __call__(self, step, x, logits):
        if step is None:
            return x

        window_start = int(0.25 * self.total_steps)
        window_end = int(0.75 * self.total_steps)
        verify_now = (
            self.revise_every is not None
            and self.revise_every > 0
            and step % self.revise_every == 0
            and window_start <= step < window_end
        )
        if not verify_now:
            return x

        generated = x[:, self.input_width :]
        generated_filled = generated != self.mask_token_id
        if not generated_filled.any():
            return x

        p = F.softmax(logits, dim=-1)
        top2_vals, _ = p.topk(2, dim=-1)
        margin = top2_vals[..., 0] - top2_vals[..., 1]

        pick_score = -margin
        candidates = torch.zeros_like(x, dtype=torch.bool)
        candidates[:, self.input_width :] = generated_filled
        if self.protect_tail > 0:
            protect_start = max(self.input_width, self.max_length - self.protect_tail)
            candidates[:, protect_start:self.max_length] = False

        verify_mask = torch.zeros_like(x, dtype=torch.bool)
        for b in range(x.shape[0]):
            cand_b = torch.nonzero(candidates[b], as_tuple=False).squeeze(-1)
            if cand_b.numel() == 0:
                continue
            m = min(int(self.candidate_m), int(cand_b.numel()))
            scores_b = pick_score[b, cand_b]
            _, idx = torch.topk(scores_b, k=m)
            verify_mask[b, cand_b[idx]] = True

        if not verify_mask.any():
            return x

        pll, x0_ver, _, x0_prob_ver, p_ver, x_filled = self._run_verify_pass(
            x, verify_mask
        )
        remask_score = self._score_from_verify(
            pll, p_ver, x0_ver, x0_prob_ver, x_filled
        )
        remask_score = torch.where(
            verify_mask, remask_score, torch.full_like(remask_score, -float("inf"))
        )

        remask_index = torch.zeros_like(x, dtype=torch.bool)
        for b in range(x.shape[0]):
            scores_b = remask_score[b]
            finite = torch.isfinite(scores_b)
            if not finite.any():
                continue
            k = min(self.remask_k, int(finite.sum().item()))
            if k <= 0:
                continue
            masked_scores = scores_b.masked_fill(~finite, float("-inf"))
            _, idx = torch.topk(masked_scores, k=k)
            remask_index[b, idx] = True

        if not remask_index.any():
            return x

        x = x.clone()
        x[remask_index] = self.mask_token_id

        if self.joint_reeval:
            # We currently only use joint re-eval as a remask-for-next-step mechanism.
            # Keeping the remasked tokens is still closer to Dream's native sampler than
            # forcing direct token substitutions in-hook.
            return x

        return x
