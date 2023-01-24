from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)


@register_criterion(
    "label_smoothed_cross_entropy_for_robust_knn_mt", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterionForRobustKnnMt(LabelSmoothedCrossEntropyCriterion):
    def get_lprobs_and_target(self, model, net_output, sample):
        # rewrite `get_lprobs_and_target` function to specify the `sample` argument 
        # of `get_normalized_probs` function
        lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
