from asyncio.log import logger
import sys
import torch

from fairseq import utils


class KnnSequenceScorer:
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        forward_hook,
        knn_search,
        recompute_distance,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.forward_hook = forward_hook
        self.knn_search = knn_search
        self.recompute_distance = recompute_distance

        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
    
    def get_knn_probs(self, target):
        # compute knn prob
            
        # T x B x C
        queries = self.forward_hook.collected_outputs[0]
        self.forward_hook.clear()

        # B x T x C
        queries = queries.transpose(0, 1)

        bsz, seq_len = queries.size()[:2]
        # B*T x C
        queries = queries.contiguous().view(-1, queries.size(-1))

        # B*T
        target = target.view(-1)
        target_padding_mask = target.ne(self.pad)

        # Reduced_B x C
        queries = queries[target_padding_mask]

        queries_device = queries.device
        queries_dtype = queries.dtype

        # Reduced_B x K
        distance, idx = self.knn_search.index.search(queries.cpu().float().numpy(), self.knn_search.cfg.num_neighbors)

        if self.recompute_distance:
            # Reduced_B X K X C
            retrieved_keys = torch.from_numpy(self.knn_search.datastore_keys[idx]).to(queries_device).type(queries_dtype)
            # Reduced_B X K
            distance = retrieved_keys.sub_(queries.unsqueeze(1)).pow_(2).sum(2)
            del retrieved_keys
        else:
            # distance and queries should have the same device
            distance = torch.from_numpy(distance).to(queries_device)
        
        del queries
        
        distance.neg_()

        if self.knn_search.datastore_value_weights is not None:
            # Reduced_B X K
            weight = torch.from_numpy(self.knn_search.datastore_value_weights[idx]).to(queries_device)
            # following the original implementation in `Efficient Nearest Neighbor Language Models`
            distance.add_(weight.log_())
            del weight

        distance = utils.softmax(distance, dim=-1)
        tgt_idx = torch.from_numpy(self.knn_search.datastore_values[idx]).to(queries_device)
        distance.masked_fill_(tgt_idx.ne(target[target_padding_mask].unsqueeze(1)), 0.0)

        # Reduced_B
        distance = distance.sum(-1)
        distance.mul_(self.knn_search.cfg.lambda_value)

        # B x T
        knn_probs = distance.new_zeros(bsz * seq_len)
        knn_probs[target_padding_mask] = distance
        knn_probs = knn_probs.view(bsz, seq_len)
        return knn_probs


    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            # B x T
            probs = probs.view(sample["target"].shape)

            # B x T
            knn_probs = self.get_knn_probs(orig_target)

            # combine lm probs and knn probs
            if len(models) == 1:
                # probs has been loged
                probs.exp_().mul_(1.0 - self.knn_search.cfg.lambda_value).add_(knn_probs).log_()
            else:
                probs.mul_(1.0 - self.knn_search.cfg.lambda_value).add_(knn_probs)
            
            del knn_probs

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
            
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "score": score_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                    }
                ]
            )
        return hypos
