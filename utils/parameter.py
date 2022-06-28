import os
import json
from utils.constants import *


class Parameter:
    def __init__(self, json_data):
        pass


class FrequencyCompressionParameters(Parameter):
    def __init__(self, json_data):
        super(FrequencyCompressionParameters, self).__init__(json_data)
        self.active = json_data["active"]
        self.type = json_data["type"]
        self.sr = json_data["sr"]
        self.f_min = json_data["f_min"]
        self.f_max = json_data["f_max"]
        self.n_freq_bins = json_data["n_freq_bins"]


class GeneralParameters(Parameter):
    def __init__(self, json_data):
        super(GeneralParameters, self).__init__(json_data)
        self.description = json_data["description"]
        self.class_name = json_data["class_name"]
        self.debug = json_data["debug"]


class CheckpointParameters(Parameter):
    def __init__(self, json_data):
        super(CheckpointParameters, self).__init__(json_data)
        self.iterations_per_landmark = json_data["iterations_per_landmark"]
        self.iterations_per_checkpoint = json_data["iterations_per_checkpoint"]
        self.checkpoints_to_keep = json_data["checkpoints_to_keep"]


class TrainingParameters(Parameter):

    def __init__(self, json_data):
        super(TrainingParameters, self).__init__(json_data)
        self.json_data = json_data
        self.base_training_directory = json_data["base_training_directory"]
        self.use_cuda = json_data["use_cuda"]
        self.learning_rate = json_data["learning_rate"]
        self.batch_size = json_data["batch_size"]
        self.beta_1 = json_data["beta_1"]
        self.beta_2 = json_data["beta_2"]
        self.iterations = json_data["iterations"]
        self.d_iterations = json_data["d_iterations"]
        self.metric_iterations = json_data["metric_iterations"]
        self.gradient_penalty_lambda = json_data["gradient_penalty_lambda"]
        self.checkpointing = CheckpointParameters(json_data["checkpointing"])


class LatentSpaceParameters(Parameter):
    def __init__(self, json_data):
        super(LatentSpaceParameters, self).__init__(json_data)
        self.dimension = json_data["latent_dimension"]
        self.latent_space_generator = json_data["latent_space_generator"]
        if json_data["latent_space_generator"] == "gaussian":
            self.data = GaussianLatentSpaceParameters(json_data["gaussian"])
        elif json_data["latent_space_generator"] == "uniform":
            self.data = UniformLatentSpaceParameters(json_data["uniform"])
        else:
            raise KeyError


class GaussianLatentSpaceParameters:
    def __init__(self, json_data=None, mean=0, std=1):
        if json_data is None:
            self.mean = mean
            self.std = std
        else:
            self.mean = json_data["mean"]
            self.std = json_data["std"]


class UniformLatentSpaceParameters:
    def __init__(self, json_data):
        self.mean = json_data["min"]
        self.var = json_data["max"]


class InversionParameters(Parameter):
    def __init__(self, json_data):
        super(InversionParameters, self).__init__(json_data)
        self.exp_power = json_data["exp_power"]


class DataParameters(Parameter):
    def __init__(self, json_data):
        super(DataParameters, self).__init__(json_data)
        self.data_directory = json_data["data_directory"]
        self.sr = json_data["sr"]
        self.n_fft = json_data["n_fft"]
        self.hop_length = json_data["hop_length"]
        self.clip_below_factor = json_data["clip_below_factor"]
        self.log_input = json_data["log_input"]
        self.n_time_bins = json_data["n_time_bins"]
        self.debug = json_data["debug"]
        self.frequency_first = json_data["frequency_first"]
        self.frequency_compression = FrequencyCompressionParameters(json_data["frequency_compression"])
        self.inversion = InversionParameters(json_data["inversion"])


class CacheParameters(Parameter):
    def __init__(self, json_data):
        super(CacheParameters, self).__init__(json_data)
        self.active = json_data["active"]
        self.cache_directory = json_data["cache_directory"]
        self.create_img = json_data["create_img"]
        self.target_ext = json_data["target_ext"]
        self.bust_cache = json_data["bust_cache"]


class Parameters:
    def __init__(self, file_name):
        with open(file_name, "r") as f:
            param_data = json.load(f)
        self.general = GeneralParameters(param_data[general])
        self.training = TrainingParameters(param_data[training])
        self.latent_space = LatentSpaceParameters(param_data[latent_space])
        self.data = DataParameters(param_data[data])
        self.cache = CacheParameters(param_data[cache])



if __name__ == '__main__':
    file = os.path.join(os.getcwd(), "../parameters.json")
    p = Parameters(file)
    print("")

