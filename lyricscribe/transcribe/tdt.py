import torch
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

from lyricscribe.transcribe.base import MAX_NEW_TOKENS, Decode, penalize_generated


def apply_controls(scores: torch.Tensor, history: list[int], decode: Decode) -> torch.Tensor:
    """
    The repetition penalty and n-gram ban every backend applies, on one row of token scores.
    *history* holds the emitted labels only, so blank is never penalised or banned.

    :param scores: ``(1, vocab)`` logits or log-probabilities.
    :param history: Labels emitted so far.
    :param decode: Decoding settings.
    :return: The processed scores.
    """
    if not history:
        return scores
    ids = torch.tensor([history], device=scores.device)
    if decode.repetition_penalty != 1.0:
        scores = penalize_generated(scores, ids, decode.repetition_penalty)
    if decode.no_repeat_ngram_size:
        scores = NoRepeatNGramLogitsProcessor(decode.no_repeat_ngram_size)(ids, scores)
    return scores


# NeMo's transcript hash for recombining identical hypotheses (batched_beam_decoding_utils).
HASH_MULTIPLIER = 6364136223846793005
HASH_INCREMENT = 1


class TDTDecoder:
    """
    Greedy and beam decoding of a TDT transducer with the shared repetition controls.

    Greedy follows transformers' TDT greedy (token and duration argmax, blank moves at least one
    frame) with NeMo's per-frame symbol cap: after ``max_symbols`` labels at one frame the pointer
    moves on. Beam search is NeMo's default TDT beam search (``BeamTDTInfer``, ``search_type``
    ``"default"``, length-normalised final ranking), ported step for step.

    :param predict: ``predict(labels (N,), state | None) -> (decoder output (N, H), state)``.
    :param joint: ``joint(encoder frames (N, H), decoder output (N, H)) -> logits (N, V + D)``,
        token logits first with blank last, then duration logits.
    :param vocab: Token logits V, blank included.
    :param blank: Blank id (``V - 1``).
    :param durations: Frames advanced per duration index.
    :param start: Label fed to the prediction network first.
    :param eos: Label that ends a row, or ``None``.
    :param max_symbols: Labels allowed at one frame before greedy moves on.
    :param max_new_tokens: Labels allowed per chunk.
    """

    def __init__(self, predict, joint, vocab: int, blank: int, durations: list[int], start: int,
                 eos: int | None = None, max_symbols: int = 10, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        """
        Keep the model callables and constants.
        """
        if blank != vocab - 1:
            raise ValueError(f"blank must be the last token logit, got {blank} of {vocab}")
        self.predict, self.joint = predict, joint
        self.vocab, self.blank, self.durations = vocab, blank, list(durations)
        self.start, self.eos = start, eos
        self.max_symbols, self.max_new_tokens = max_symbols, max_new_tokens

    @torch.inference_mode()
    def greedy(self, enc: torch.Tensor, lengths: torch.Tensor, decode: Decode = Decode()) -> list[list[int]]:
        """
        Batched greedy decoding.

        :param enc: ``(B, T, H)`` projected encoder output.
        :param lengths: ``(B,)`` valid frames per row.
        :param decode: Repetition controls (``num_beams`` must be 1).
        :return: Emitted labels per row.
        """
        batch, frames = enc.shape[0], enc.shape[1]
        rows = torch.arange(batch, device=enc.device)
        durations = torch.tensor(self.durations, device=enc.device)
        controlled = bool(decode.active() - {"num_beams"})
        frame = torch.zeros(batch, dtype=torch.long, device=enc.device)
        at_frame = torch.zeros_like(frame)
        state = None
        dec, state = self.predict(torch.full((batch,), self.start, dtype=torch.long, device=enc.device), state)
        out = [[] for _ in range(batch)]
        done = lengths <= 0
        for _ in range(self.max_symbols * frames - 1):
            logits = self.joint(enc[rows, frame.clamp(max=frames - 1)], dec)
            token_logits = logits[:, : self.vocab].float()
            if controlled:
                token_logits = torch.cat([apply_controls(token_logits[b:b + 1], out[b], decode) for b in range(batch)])
            token = token_logits.argmax(-1)
            duration = durations[logits[:, self.vocab:].argmax(-1)]
            blank = token == self.blank
            duration = torch.where(blank & (duration == 0), torch.ones_like(duration), duration)
            at_frame = torch.where(~blank & (duration == 0), at_frame + 1, torch.zeros_like(at_frame))
            force = (at_frame >= self.max_symbols).long()
            at_frame = at_frame * (1 - force)
            duration = duration + force
            for b in torch.nonzero(~done & ~blank).flatten().tolist():
                out[b].append(int(token[b]))
            frame = frame + duration
            ended = (frame >= lengths) | torch.tensor([len(o) >= self.max_new_tokens for o in out], device=enc.device)
            if self.eos is not None:
                ended |= token == self.eos
            done = done | ended
            if bool(done.all()):
                break
            if bool((~blank).any()):
                new_dec, new_state = self.predict(token, state)
                keep = ~blank
                dec = torch.where(keep[:, None], new_dec, dec)
                state = tuple(torch.where(keep[None, :, None], n, o) for n, o in zip(new_state, state))
        return out

    @torch.inference_mode()
    def beam(self, enc: torch.Tensor, lengths: torch.Tensor, beam: int, decode: Decode = Decode()) -> list[list[int]]:
        """
        NeMo's ``malsd_batch`` TDT beam search, with the repetition controls applied to each
        hypothesis's token log-probabilities before the top-k.

        :param enc: ``(B, T, H)`` projected encoder output.
        :param lengths: ``(B,)`` valid frames per row.
        :param beam: Beam width.
        :param decode: Repetition controls.
        :return: The best hypothesis's labels per row.
        """
        batch, device = enc.shape[0], enc.device
        num_d = len(self.durations)
        durations = torch.tensor(self.durations, device=device)
        controlled = bool(decode.active() - {"num_beams"})
        rows = torch.arange(batch, device=device)[:, None].expand(batch, beam)
        parent_of = torch.arange(beam, device=device)[None, :, None].expand(batch, beam, beam).reshape(batch, -1)
        scores = torch.full((batch, beam), float("-inf"), device=device, dtype=enc.dtype)
        scores[:, 0] = 0.0
        next_time = torch.zeros((batch, beam), dtype=torch.long, device=device)
        at_frame = torch.zeros_like(next_time)
        n_labels = torch.zeros_like(next_time)
        last_label = torch.full_like(next_time, -1)
        hashes = torch.zeros_like(next_time)
        history = [[[] for _ in range(beam)] for _ in range(batch)]
        last = (lengths - 1)[:, None].expand(batch, beam)
        safe_time = torch.zeros_like(next_time)
        active = next_time <= last
        dec, state = self.predict(torch.full((batch * beam,), self.start, dtype=torch.long, device=device), None)
        while bool(active.any()):
            logits = self.joint(enc[rows.reshape(-1), safe_time.reshape(-1)], dec)
            log_probs = torch.log_softmax(logits[:, : self.vocab], dim=-1).view(batch, beam, -1)
            dur_lp = torch.log_softmax(logits[:, self.vocab:], dim=-1).view(batch, beam, -1)
            if controlled:
                for b, k in torch.nonzero(active).tolist():
                    if history[b][k]:
                        log_probs[b, k] = apply_controls(log_probs[b, k][None], history[b][k], decode)[0]
            capped = n_labels >= self.max_new_tokens
            if bool(capped.any()):
                log_probs[..., :-1] = torch.where(capped[..., None], float("-inf"), log_probs[..., :-1])
            total = log_probs[:, :, :, None] + dur_lp[:, :, None, :]
            top_lp, top_idx = torch.topk(total.view(batch, beam, -1), beam, dim=-1, largest=True, sorted=True)
            labels = top_idx // num_d
            dur_idx = top_idx % num_d
            dur_idx = torch.where((labels == self.blank) & (durations[dur_idx] == 0), 1, dur_idx)
            blank_lp = log_probs[..., -1] + dur_lp.max(dim=-1).values
            cand = scores[..., None] + top_lp
            forced = scores + blank_lp
            cand = torch.where(active[..., None], cand, float("-inf"))
            cand[..., 0] = torch.where(active, cand[..., 0], scores)
            labels = torch.where(active[..., None], labels, -1)
            force = (at_frame >= self.max_symbols) & active
            cand = torch.where(force[..., None], float("-inf"), cand)
            cand[..., 0] = torch.where(force, forced, cand[..., 0])
            labels = torch.where(force[..., None], self.blank, labels)
            dur_idx = torch.where(force[..., None] & (durations[dur_idx] == 0), 1, dur_idx)
            new_scores, picked = torch.topk(cand.view(batch, -1), k=beam, largest=True, sorted=True)
            parent = torch.gather(parent_of, 1, picked)
            new_labels = torch.gather(labels.reshape(batch, -1), 1, picked)
            new_dur = durations[torch.gather(dur_idx.reshape(batch, -1), 1, picked)]
            # NeMo's add_results_no_checks_ for TDT.
            extended = new_labels >= 0
            with_label = extended & (new_labels != self.blank)
            prev_last = torch.gather(last_label, 1, parent)
            steps = torch.gather(next_time, 1, parent)
            new_dur = torch.where(extended, new_dur, 0)
            next_time = torch.where(extended, steps + new_dur, steps)
            at_frame = torch.where(extended & (new_dur > 0), 0, torch.gather(at_frame, 1, parent) + with_label)
            n_labels = torch.gather(n_labels, 1, parent) + with_label
            scores = new_scores
            prev_hash = torch.gather(hashes, 1, parent)
            hashes = torch.where(with_label, prev_hash * HASH_MULTIPLIER + HASH_INCREMENT + new_labels, prev_hash)
            last_label = torch.where(with_label, new_labels, prev_last)
            parents, added, adds = parent.tolist(), new_labels.tolist(), with_label.tolist()
            history = [[history[b][parents[b][k]] + ([added[b][k]] if adds[b][k] else []) for k in range(beam)]
                       for b in range(batch)]
            # NeMo's recombine_hyps_: identical hypotheses keep one entry with the summed probability.
            if beam > 1:
                same = ((hashes[:, :, None] == hashes[:, None, :]) & (last_label[:, :, None] == last_label[:, None, :])
                        & (n_labels[:, :, None] == n_labels[:, None, :]) & (next_time[:, :, None] == next_time[:, None, :]))
                matrix = torch.where(same, scores[:, None, :].expand(batch, beam, beam), float("-inf"))
                keep = torch.arange(beam, device=device)[None, :] == matrix.argmax(-1)
                scores = torch.where(keep, torch.logsumexp(matrix, dim=-1).to(scores.dtype), float("-inf"))
            feed = torch.where(new_labels >= 0, new_labels, self.blank)
            preserve = (feed == self.blank).reshape(-1)
            flat_parent = (parent + torch.arange(batch, device=device)[:, None] * beam).reshape(-1)
            prev_dec = dec[flat_parent]
            prev_state = tuple(x[:, flat_parent] for x in state)
            new_dec, new_state = self.predict(feed.reshape(-1), prev_state)
            dec = torch.where(preserve[:, None], prev_dec, new_dec)
            state = tuple(torch.where(preserve[None, :, None], p, n) for p, n in zip(prev_state, new_state))
            safe_time = torch.minimum(next_time, last)
            active = next_time <= last
        order = torch.sort(scores / (n_labels.to(scores.dtype) + 1), dim=-1, descending=True).indices[:, 0].tolist()
        return [history[b][order[b]] for b in range(batch)]
