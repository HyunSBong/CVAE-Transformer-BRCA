import torch


def PartialLogLikelihood(logits, durations, events):
    """Compute the partial log likelihood as defined in DeepSurv

    Args:
        logits (Tensor): risk function, model output (1D)
        events (Tensor): Event indicator, 1 = death, 0=censored (1D)
        durations (Tensor): failure event time (1D)

    Returns:
        Tensor: [description]

    Risk set: R = {T_j >= T_i}
    Risk set approximation with cumsum on sorted tensor
    """
    # sort risk and events based on decreasing time
    idx = torch.argsort(durations, descending=True)
    logits = logits[idx].view(-1)
    events = events[idx]

    # not really the Risk set but good approximation
    # from pytorch doc The computation is numerically stabilized.
    partial_L = logits.logcumsumexp(dim=0)
    return (-1.0 * (logits - partial_L).mul(events).sum()) / events.sum()
