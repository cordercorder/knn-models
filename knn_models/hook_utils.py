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
