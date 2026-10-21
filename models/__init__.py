"""Models for inference with the published checkpoint configurations."""


def build_model(config):
    if config["model"] == "crossvivit":
        from .crossvivit import RoCrossViViT

        return RoCrossViViT(**config["model_kwargs"])
    if config["model"] == "lstm-att-sat":
        from .lstm_att_sat import Solar_Predictor

        return Solar_Predictor(**config["model_kwargs"])
    raise ValueError(f"Unknown model: {config['model']}")
