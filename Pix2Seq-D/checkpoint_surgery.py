import torch


def perform_checkpoint_surgery(ckpt):
    '''
        Performs checkpoint surgery on the given checkpoint file.
        The surgery consists of replicating the weights of the first and last convolutional layer
            because Cityscapes target images have 1 channel while KittiSTEP has 3.

        :param ckpt: The checkpoint file to perform surgery on (comes from pretraining on Cityscapes).
        :return: The checkpoint file with the surgery performed.
    '''

    # Adapt first convolutional layer
    
    # UNet forward pass uses
    # x = torch.cat((conditioning, x_self_cond, embedding, x), dim=1)
    # with channels = 1, BITS = 8, embedding_size = 32, frames_before = 5
    # conditioning is of size frames_before*BITS*channels = 5*8*1
    # x_self_cond is of size BITS*channels = 8*1
    # embedding is of size embedding_size = 32
    # x is of size BITS*channels = 8*1
    
    chunks = list(torch.tensor_split(
        ckpt["decoder.model.init_conv.weight"],
        [5*8, 5*8+8, 5*8+8+32],
        dim=1
    ))

    
    chunks_to_expand = [0, 1, 3]

    for i in chunks_to_expand:
        chunks[i] = chunks[i].repeat(1, 3, 1, 1)

    ckpt["decoder.model.init_conv.weight"] = torch.cat(chunks, dim=1)

    # Adapt last convolutional layer
    ckpt["decoder.model.final_conv.weight"] = ckpt["decoder.model.final_conv.weight"].repeat(
        3, 1, 1, 1)
    ckpt["decoder.model.final_conv.bias"] = ckpt["decoder.model.final_conv.bias"].repeat(
        3)

    return ckpt

def perform_checkpoint_surgery_ema(ckpt):
    '''
        Performs checkpoint surgery on the given checkpoint file.
        The surgery consists of replicating the weights of the first and last convolutional layer
            because Cityscapes target images have 1 channel while KittiSTEP has 3.

        :param ckpt: The checkpoint file to perform surgery on (comes from pretraining on Cityscapes).
        :return: The checkpoint file with the surgery performed.
    '''

    # Adapt first convolutional layer
    
    # UNet forward pass uses
    # x = torch.cat((conditioning, x_self_cond, embedding, x), dim=1)
    # with channels = 1, BITS = 8, embedding_size = 32, frames_before = 5
    # conditioning is of size frames_before*BITS*channels = 5*8*1
    # x_self_cond is of size BITS*channels = 8*1
    # embedding is of size embedding_size = 32
    # x is of size BITS*channels = 8*1
    
    for model in ["ema_model", "online_model"]:    
        chunks = list(torch.tensor_split(
            ckpt[f"{model}.decoder.model.init_conv.weight"],
            [5*8, 5*8+8, 5*8+8+32],
            dim=1
        ))


        chunks_to_expand = [0, 1, 3]

        for i in chunks_to_expand:
            chunks[i] = chunks[i].repeat(1, 3, 1, 1)

        ckpt[f"{model}.decoder.model.init_conv.weight"] = torch.cat(chunks, dim=1)

        # Adapt last convolutional layer
        ckpt[f"{model}.decoder.model.final_conv.weight"] = ckpt[f"{model}.decoder.model.final_conv.weight"].repeat(
            3, 1, 1, 1)
        ckpt[f"{model}.decoder.model.final_conv.bias"] = ckpt[f"{model}.decoder.model.final_conv.bias"].repeat(
            3)

    return ckpt
