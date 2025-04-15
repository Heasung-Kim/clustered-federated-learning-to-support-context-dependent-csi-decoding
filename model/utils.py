def get_model_class(config):
    model_class = None
    if "VQCINet" in config["model"]:
        if config["model"] == "VQCINet64":
            from model.neural_networks.vqcinet.vqcinet import VQCINet64
            model_class = VQCINet64
        elif config["model"] == "VQCINet128":
            from model.neural_networks.vqcinet.vqcinet import VQCINet128
            model_class = VQCINet128
        elif config["model"] == "VQCINet256":
            from model.neural_networks.vqcinet.vqcinet import VQCINet256
            model_class = VQCINet256

    return model_class
