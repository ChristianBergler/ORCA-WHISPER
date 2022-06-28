# ORCA-WHISPER (An Automatic Killer Whale Sound Type Generation Toolkit Using Deep Learning)


# Installation
## LTFAT (Large Time Frequency Analysis Toolbox)
- It is recommended to install this package **globally** and to follow the installation instructions
provided by the devloper [Here](http://dev.pages.lis-lab.fr/ltfatpy/install.html)

# Configuration
The configuration of ORCA-WHISPER is controlled through the [parameters.json](./parameters.json) file in the root directory


# Training
- Set the desired parameters in [parameters.json](./parameters.json)
- Run [main.py](./main.py)

# Sound Sample Generation
- After you are satisfied with the result produced by the generator, initialize the `GANSampler` class in [generate.py](./generate.py)
```python
    parameter_file = "parameters.json"
    parameters = Parameters(parameter_file)
    g_inverter = GaborInverter(parameters.data, device=torch.device("cpu"))
    name_gen = OrcaSpotNameGenerator()
    checkpoint_path = "ORCA.pt"
    output_dir = "output"
    class_name = "ORCA"
    sample_count = 100
    noise_generator = GaussianNoiseGenerator(parameters=GaussianLatentSpaceParameters(),
                                                      batch_size=1,
                                                      latent_dimension_size=100)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    GANSampler(
            name_generator=name_gen,
            class_name=class_name,
            checkpoint_path=checkpoint_path,
            noise_generator=noise_generator,
            device=torch.device("cpu"),
            f_min=0,
            f_max=10000,
            inverter=g_inverter,
            output_base=output_dir)
    )(sample_count)
```