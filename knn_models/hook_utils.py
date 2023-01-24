import os
import torch
import logging

from knn_models.dim_reduce_utils import (
    CompactNet,
    PCATransform, 
)


logger = logging.getLogger(__name__)


class ForwardHook:
    """Hook for collecting the output of specified module in TransformerDecoder"""
    def __init__(self):
        self.collected_outputs = []
    
    def forward_hook_function(self, module, input, output):
        # assume the output is always tuple or tensor
        if isinstance(output, tuple):
            collected_output = output[0].detach()
        else:
            collected_output = output.detach()
        
        self.collected_outputs.append(collected_output)

    def clear(self):
        self.collected_outputs.clear()


class DimReduceForwardHook(ForwardHook):
    """"similar to ForwardHook while applying additional transformation to the collected output"""
    def __init__(self, args):
        super().__init__()

        if args.dim_reduce_method == "PCA":
            transform = PCATransform(**args)
        elif args.dim_reduce_method == "PCKMT":
            transform = CompactNet(**args)
        else:
            raise ValueError("Unkown dimension reduction method")
        
        transform_ckpt_path = os.path.join(args.datastore, args.transform_ckpt_name)

        logger.info(f"Loading transformation from {transform_ckpt_path}")
        ckpt = torch.load(transform_ckpt_path, map_location="cpu")
        transform.load_state_dict(ckpt)

        use_cuda = torch.cuda.is_available() and not args.cpu
        if use_cuda:
            logger.info(f"Moving {transform.__class__.__name__} to GPU")
            transform = transform.cuda()
        
        transform.eval()

        self.transform = transform
    
    def forward_hook_function(self, module, input, output):
        # assume the output is always tuple or tensor
        if isinstance(output, tuple):
            collected_output = output[0].detach()
        else:
            collected_output = output.detach()
        
        with torch.no_grad():
            collected_output = self.transform(collected_output)

        self.collected_outputs.append(collected_output)
