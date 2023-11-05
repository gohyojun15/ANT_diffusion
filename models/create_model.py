from models.DiT.models import DiT_models


def create_model(model_config):
    """
    Create various architectures from model_config
    """
    model = DiT_models[model_config.name](
        input_size=model_config.param.latent_size, num_classes=model_config.param.num_classes
    )
    return model
