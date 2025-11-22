from segmentation_models_pytorch.losses import TverskyLoss
from torch.nn import BCELoss 
from .WeightedCrossEntropyLoss import WeightedCrossEntropyLoss
from .BBoxAwareLoss import BBoxAwareLoss

__all__ = [
    "WeightedCrossEntropyLoss",
    "BBoxAwareLoss",
    "TverskyLoss",
    "BCELoss",
    "get_loss_registry"
]

def get_loss_registry() -> dict[str, type]:
    """Returns a mapping of loss class name to the loss class.

    Only names that are listed in ``__all__`` and are classes available
    in the module globals are included in the returned registry.
    """
    import inspect
    return {
        name: obj
        for name in __all__
        if (obj := globals().get(name)) and inspect.isclass(obj)
    }